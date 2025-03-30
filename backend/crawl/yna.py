import requests
from bs4 import BeautifulSoup
from datetime import datetime


def get_data():
    url = 'https://www.yna.co.kr/news?site=navi_latest_depth01'
    response = requests.get(url)

    # 연도 정보 부재로 현재 연도로 대체(수정 필요)
    current_year = datetime.now().year

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # ul.list의 li 요소를 선택
        news_list = soup.select('#container > div.container521 > div.content03 > div.section01 > section > div > ul > li')

        result = []
        for news in news_list:
            # 제목 (span 안의 텍스트)
            title_tag = news.select_one('div > div > strong > a > span')
            if title_tag:
                title = title_tag.text.strip()
            else:
                continue

            # 링크 (a 태그의 href)
            link_tag = news.select_one('div > div > strong > a')
            if link_tag and link_tag.attrs.get('href'):
                link = link_tag.attrs['href']
            else:
                continue

            # 날짜 (시간 정보)
            date_tag = news.select_one('div > div > span')
            if date_tag:
                # 시간 (예: 10-30 18:00)
                date_text = date_tag.text.strip()
                # 10-30 -> 2024.10.30
                date_unformatted = date_text.split(' ')[0]
                date = f"{current_year}.{date_unformatted.replace('-', '.')}"
            else:
                continue

            # 결과에 추가
            dict_data = {"title": title, "link": link, "date": date}
            result.append(dict_data)

        return {"연합뉴스": result}
    else:
        print(f"HTTP 요청 실패: {response.status_code}")
        return {"Error": response.status_code}
