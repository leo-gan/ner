from pathlib import Path


class ResultCode:
    SUCCESS = 0
    ERROR = 1


def evaluate(evaluation_file_path: Path, result_output_dir: Path) -> ResultCode:
    # Mock evaluation logic
    return ResultCode.SUCCESS
