import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from typing import List, Dict
from config.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, NEWS_FRESHNESS_DAYS

class NewsFetcher:
    def __init__(self):
        self.connection = None
        self.connect()
    
    def connect(self):
        """데이터베이스 연결"""
        try:
            self.connection = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            print(f"데이터베이스 연결 성공: {DB_HOST}:{DB_PORT}/{DB_NAME}")
        except psycopg2.Error as err:
            print(f"데이터베이스 연결 실패: {err}")
            raise
    
    def get_news_titles(self, days: int = None) -> List[Dict]:
        """
        뉴스 제목들을 가져옵니다.
        기본적으로 7일 이내의 뉴스를 가져옵니다.
        """
        if not self.connection or self.connection.closed:
            self.connect()
        
        if days is None:
            days = NEWS_FRESHNESS_DAYS
        
        # 지정된 일수 전 날짜 계산
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = """
        SELECT title 
        FROM newsinfo 
        WHERE date >= %s 
        ORDER BY date DESC
        """
        
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, (cutoff_date,))
            results = cursor.fetchall()
            cursor.close()
            
            print(f"뉴스 {len(results)}개 가져옴 ({days}일 이내)")
            return results
            
        except psycopg2.Error as err:
            print(f"쿼리 실행 실패: {err}")
            return []
    
    def close(self):
        """데이터베이스 연결 종료"""
        if self.connection and not self.connection.closed:
            self.connection.close()
            print("데이터베이스 연결 종료")

def fetch_news_from_db() -> List[Dict]:
    """
    데이터베이스에서 뉴스를 가져오는 편의 함수
    """
    fetcher = NewsFetcher()
    try:
        news_data = fetcher.get_news_titles()
        return news_data
    finally:
        fetcher.close()

if __name__ == "__main__":
    # 테스트 코드
    fetcher = NewsFetcher()
    try:
        news = fetcher.get_news_titles()
        print(f"가져온 뉴스 수: {len(news)}")
        if news:
            print("샘플 뉴스:")
            for i, item in enumerate(news[:3]):
                print(f"{i+1}. {item['title']}")
    finally:
        fetcher.close()
