const axios = require('axios');
const cheerio = require('cheerio');

// 웹사이트 URL 설정
const url = 'https://www.incheon.go.kr/IC010205';

async function crawlIncheonNews() {
    try {
        const { data } = await axios.get(url);
        const $ = cheerio.load(data);

        // 뉴스 데이터 선택
        const newsList = $('.board-article-group');
        const result = [];

        newsList.each((_, element) => {
            const news = $(element);

            // 링크
            const linkTag = news.closest('a');
            let link = '';
            if (linkTag.length) {
                link = 'https://www.incheon.go.kr' + linkTag.attr('href');
            }

            // 제목
            const titleTag = news.find('.subject');
            const title = titleTag.length ? titleTag.text().trim() : '';

            // 날짜
            const dateTag = news.find("dt:contains('제공일자')");
            let date = '';
            if (dateTag.length) {
                date = dateTag.next('dd').text().trim().replace(/-/g, '.');
            }

            // 결과 추가
            result.push({ title, link, date });
        });

        console.log({ "인천광역시보도자료": result });
        return result;
    } catch (error) {
        console.error('Error fetching the page:', error);
        return [];
    }
}

// 약 1초
// crawlIncheonNews();
module.exports = crawlIncheonNews;
