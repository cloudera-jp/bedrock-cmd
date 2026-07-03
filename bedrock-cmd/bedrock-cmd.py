"""Helper utilities for working with Amazon Bedrock from Python notebooks"""
# Python Built-Ins:
import os
import sys
import json
import re
from typing import Optional

# External Dependencies:
import boto3
from botocore.config import Config

# global
modelId = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
accept = 'application/json'
contentType = 'application/json'
system__prompt = "あなたは、関西電力の文書ファイルを、機能階層CSVで定義される階層情報によってラベル付を行う、文書ファイル管理のエキスパートです。"

def get_bedrock_client(
    assumed_role: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    region: Optional[str] = None,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    assumed_role :
        Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
        specified, the current active credentials will be used.
    endpoint_url :
        Optional override for the Bedrock service API Endpoint. If setting this, it should usually
        include the protocol i.e. "https://..."
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    """
    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    # print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        # print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )
        print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    bedrock_client = session.client(
        service_name="bedrock-runtime",
        config=retry_config,
        **client_kwargs
    )

    # print("boto3 Bedrock client successfully created!")
    # print(bedrock_client._endpoint)
    return bedrock_client



# Setting up the prompt syntax for the corresponding model
def prompt_construction(content):
    full_prompt = [
      {
        "role": "user", 
        "content": f"{content}"
      }
    ]
  
    return full_prompt

# Setting up the API call in the correct format for the corresponding model
def json_format(tokens, temperature, top_p, full_prompt):
    body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": tokens,
            "system": system__prompt,
            "messages": full_prompt,
            "temperature": temperature,
            "top_p": top_p
            }, ensure_ascii=False)
    
    # print(f"Debug Body: {body}")
  
    return body


if __name__ == "__main__":
    # コマンドライン引数の数をチェック
    # if len(sys.argv) != 3:
    #     print("Usage: python script.py <input_file_path> <page_number>")
    #     sys.exit(1)

    # コマンドライン引数を取得
    # input_file = sys.argv[1]
    # try:
    #     page_number = int(sys.argv[2])
    # except ValueError:
    #     print("Error: Page number must be an integer.")
    #     sys.exit(1)

    # 関数を実行

    # print("Main Module Started!")

    # Initializing the bedrock client using AWS credentials
    boto3_bedrock = get_bedrock_client(
        region=os.environ.get("AWS_DEFAULT_REGION", None))

    # 標準入力から全データを読み込み、変数に代入
    input_prompt = sys.stdin.read()
    
    # プロンプトを構築
    full_prompt = prompt_construction(input_prompt)
    # LLMのパラメータ設定
    max_tokens = 2000
    temperature = 0.00
    top_p = 0
    # bedrock apiのbody作成
    body = json_format(max_tokens, temperature, top_p, full_prompt)

    # Foundation model is invoked here to generate a response
    response = boto3_bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    result = response_body['content'][0]['text']
    # print(result)

    # 正規表現を使用してJSONブロックを抽出
    json_block = re.search(r'```json\n(.*?)\n```', result, re.DOTALL)

    if json_block:
        json_content = json_block.group(1)
        print(json_content)
    else:
        print("JSONブロックが見つかりませんでした。")

    # print("End boto3_bedrock")


