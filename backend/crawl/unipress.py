import requests
import re
from bs4 import BeautifulSoup

def get_data():
    url = 'http://www.unipress.co.kr/news/articleList.html?sc_section_code=S1N1&view_type=sm'

    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        news_list = soup.select('div.list-block')

        result = []
        for news in news_list:
            title = news.select_one('div.list-titles a strong').text.strip()

            # 대학핫뉴스-일반대 | 배지우 | 2024-10-30 15:46
            date_text = news.select_one('div.list-dated').text
            # 2024-10-30
            match = re.search(r'(\d{4})-(\d{2})-(\d{2})', date_text)
            if match is not None:
                # 2024.10.30
                date = f"{match.group(1)}.{match.group(2)}.{match.group(3)}"

            link = "http://www.unipress.co.kr" + news.select_one('div.list-titles a').attrs['href']

            #print(f"제목: {title}")
            #print(f"날짜: {date}")
            #print(f"링크: {link}")
            #print("-" * 100)
            dict_data = {"title": title, "link": link, "date": date}
            result.append(dict_data)

        return {"대학지성IN&OUT": result}
    else:
        print(f"Failed to fetch the page, status code: {response.status_code}")
        return {"Error": response.status_code}