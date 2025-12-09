"""
Crawlers Package Initialization
======================================================================
설명: 크롤러 클래스들을 외부로 노출합니다.
참고: 숫자로 시작하는 파일명(01_...)은 동적으로 임포트하여 연결합니다.
======================================================================
"""

import importlib
from .base_crawler import BaseCrawler

__all__ = ["BaseCrawler"]

# 동적 임포트 매핑 (파일명: 클래스명)
_DYNAMIC_MODULES = {
    "01_crawling_historydb": "HistoryDBCrawler",
    "01_crawling_nrich": "NRICHCrawler",
    "01_crawling_kyu": "KyudbCrawler"
}

for _mod_name, _class_name in _DYNAMIC_MODULES.items():
    try:
        # 현재 패키지(.) 내에서 모듈 임포트
        _module = importlib.import_module(f".{_mod_name}", package=__name__)
        _class = getattr(_module, _class_name)
        
        # 전역 네임스페이스에 클래스 등록
        globals()[_class_name] = _class
        __all__.append(_class_name)
        
    except (ImportError, AttributeError):
        # 파일이 아직 없거나 클래스명이 다를 경우 무시
        pass