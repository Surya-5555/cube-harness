import json
import requests

code = """
def add_one(x):
    return x + 1

assert add_one(1) == 2
assert add_one(5) == 8
"""


def test_sandbox(sandbox_fusion_base_url: str):
    compile_timeout = 10
    run_timeout = 10
    request_timeout = 10
    stdin = "15"
    memory_limit_mb = 1024
    sandbox_fusion_url = sandbox_fusion_base_url + "/run_code"
    language = "python"

    payload = json.dumps(
        {
            "compile_timeout": compile_timeout,
            "run_timeout": run_timeout,
            "code": code,
            "stdin": stdin,
            "memory_limit_MB": memory_limit_mb,
            "language": language,  # Use the passed language parameter
            "files": {},
            "fetch_files": [],
        }
    )

    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    try:
        requests.post(
            sandbox_fusion_url,
            headers=headers,
            data=payload,
            timeout=request_timeout,  # Use the calculated timeout
        )
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to sandbox: {e}")
        return False

    return True
