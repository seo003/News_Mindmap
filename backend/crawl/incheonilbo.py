import requests
from bs4 import BeautifulSoup

def get_data():
    url = 'https://www.incheonilbo.com/news/articleList.html?sc_sub_section_code=S2N14&view_type=sm'
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # 원하는 ul > li들 선택
        news_items = soup.select('#section-list ul > li')
        result = []
        # print(news_items)
        for item in news_items:
            # 제목 및 링크
            title_tag = item.select_one('h2.titles > a')
            if not title_tag:
                continue
            base_url = "https://www.incheonilbo.com"
            title = title_tag.get_text(strip=True)
            link = base_url + title_tag.get('href', '')

            # 날짜
            byline_ems = item.select('span.byline > em')
            full_date = byline_ems[2].get_text(strip=True)
            # 날짜만 추출
            date = full_date.split()[0]

            dict_data = {"title": title, "link": link, "date": date}
            result.append(dict_data)
        return result

    else:
        print(f"Failed to fetch the page, status code: {response.status_code}")
        return ['error']
