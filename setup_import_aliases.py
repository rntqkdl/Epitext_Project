#!/usr/bin/env python3
"""
epitext 프로젝트에서 편리한 모듈 임포트를 지원하는 도우미 스크립트.

이 스크립트는 연구용 저장소 루트에서 실행되며 다음 작업을 수행합니다.

1. `epitext_data`와 `epitext_model` 디렉터리를 생성하여 `1_data`와
   `3_model` 디렉터리에 있는 서브패키지들을 동적으로 로드할 수 있는
   alias 패키지를 만듭니다.
2. `1_data/__init__.py`와 `3_model/__init__.py`에 안내 메시지를 추가하여
   alias 패키지 사용법을 설명합니다.

사용법:

```
cd Epitext_Project
python setup_import_aliases.py
```

실행 후 `import epitext_data.crawlers`나 `import epitext_model.nlp`와 같이
직관적인 모듈 임포트가 가능합니다.
"""

import os
import importlib.util
import textwrap


def create_epitext_data_package(base_dir: str) -> None:
    """`epitext_data` alias 패키지를 생성합니다."""
    alias_dir = os.path.join(base_dir, "epitext_data")
    os.makedirs(alias_dir, exist_ok=True)
    content = textwrap.dedent(
        """
        # -*- coding: utf-8 -*-
        """epitext_data: 1_data 모듈에 대한 alias 패키지.

        이 패키지를 사용하면 `1_data` 디렉터리의 서브패키지들을
        파이썬 이름 규칙에 맞게 불러올 수 있습니다. 예를 들어,

            from epitext_data.crawlers.pdf_processing import main as pdf_main
            pdf_main()

        처럼 사용할 수 있습니다.
        """

        import importlib.util
        import os

        _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _DATA_DIR = os.path.join(_PROJECT_ROOT, "1_data")

        def _load_package(name: str):
            subdir = os.path.join(_DATA_DIR, name)
            init_file = os.path.join(subdir, "__init__.py")
            spec = importlib.util.spec_from_file_location(f"epitext_data.{name}", init_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        # 동적으로 서브패키지를 로드하여 속성으로 노출합니다.
        crawlers = _load_package("crawlers")
        preprocess = _load_package("preprocess")
        eda = _load_package("eda")
        utils = _load_package("utils")
        """
    ).lstrip()
    with open(os.path.join(alias_dir, "__init__.py"), "w", encoding="utf-8") as f:
        f.write(content)


def create_epitext_model_package(base_dir: str) -> None:
    """`epitext_model` alias 패키지를 생성합니다."""
    alias_dir = os.path.join(base_dir, "epitext_model")
    os.makedirs(alias_dir, exist_ok=True)
    content = textwrap.dedent(
        """
        # -*- coding: utf-8 -*-
        """epitext_model: 3_model 모듈에 대한 alias 패키지.

        이 패키지를 사용하면 `3_model` 디렉터리의 서브패키지들을
        파이썬 이름 규칙에 맞게 불러올 수 있습니다. 예를 들어,

            from epitext_model.nlp.sikuroberta.train import train_task
            train_task()

        처럼 사용할 수 있습니다.
        """

        import importlib.util
        import os

        _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _MODEL_DIR = os.path.join(_PROJECT_ROOT, "3_model")

        def _load_package(name: str):
            subdir = os.path.join(_MODEL_DIR, name)
            init_file = os.path.join(subdir, "__init__.py")
            spec = importlib.util.spec_from_file_location(f"epitext_model.{name}", init_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        # 서브패키지 노출
        nlp = _load_package("nlp")
        vision = _load_package("vision")
        """
    ).lstrip()
    with open(os.path.join(alias_dir, "__init__.py"), "w", encoding="utf-8") as f:
        f.write(content)


def append_alias_note_to_init(file_path: str, alias_name: str) -> None:
    """기존 __init__.py 파일에 alias 패키지 사용법 안내 메시지를 추가합니다."""
    if not os.path.exists(file_path):
        return
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    note = (
        f"\n\nNote: 이 디렉터리를 직접 import하는 대신 `epitext_{alias_name}` alias 패키지를 사용하여 "
        f"서브모듈을 불러올 수 있습니다. 예: `import epitext_{alias_name}`.\n"
    )
    if note not in content:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(note)


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 1. alias 패키지 생성
    create_epitext_data_package(base_dir)
    create_epitext_model_package(base_dir)
    # 2. __init__.py에 안내 추가
    append_alias_note_to_init(os.path.join(base_dir, "1_data", "__init__.py"), "data")
    append_alias_note_to_init(os.path.join(base_dir, "3_model", "__init__.py"), "model")
    print("epitext_data 및 epitext_model 패키지가 생성되었습니다. 이제 import를 통해 사용할 수 있습니다.")


if __name__ == "__main__":
    main()