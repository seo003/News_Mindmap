const axios = require('axios');
const cheerio = require('cheerio');

async function crawlUsline() {
    try {
        const url = 'https://www.usline.kr/news/articleList.html?view_type=sm';

        // 요청 헤더 설정
        const headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
            'Referer': 'https://www.google.com/', // 검색을 통해 접근한 것처럼 보이게 설정
        };

        // 웹페이지 요청
        const { data } = await axios.get(url, { headers });

        // HTML 파싱
        const $ = cheerio.load(data);
        const newsList = $('section#section-list div.view-cont');

        const result = [];

        newsList.each((_, el) => {
            const titleTag = $(el).find('h4 a');
            // 제목 및 링크크
            let title = titleTag.text().trim();
            let link = 'https://www.usline.kr' + titleTag.attr('href');

            // 날짜
            let date = null;
            $(el).find('em').each((_, em) => {
                const match = $(em).text().trim().match(/(\d{4}\.\d{2}\.\d{2}) \d{2}:\d{2}/);
                if (match) {
                    date = match[1];
                    return false; // 첫 번째 일치하는 날짜만 사용
                }
            });

            result.push({ title, link, date });
        });

        console.log({ "유스라인(Usline)": result });
        return result;
    } catch (error) {
        console.error('크롤링 실패:', error.message);
        return [];
    }
}

// 약 0초
// crawlUsline();
module.exports = crawlUsline;
