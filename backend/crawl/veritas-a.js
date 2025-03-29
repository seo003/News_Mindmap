const axios = require('axios');
const cheerio = require('cheerio');

async function crawlVeritas() {
    try {
        const url = 'https://www.veritas-a.com/news/articleList.html?sc_section_code=&view_type=sm';

        // 요청 헤더 설정
        const headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
            'Referer': 'https://www.google.com/', // 검색을 통해 접근한 것처럼 보이게 설정
        };

        // 웹페이지 요청
        const { data } = await axios.get(url, { headers });

        // HTML 로드
        const $ = cheerio.load(data);

        // 뉴스 목록
        const newsList = $('#section-list > ul > li');

        const result = [];

        newsList.each((index, news) => {
            // 제목 및 링크
            const titleTag = $(news).find('div h2 a');
            const writeInfo = $(news).find('.byline');

            if (titleTag.length && writeInfo.length) {
                const title = titleTag.text().trim();
                const link = `https://www.veritas-a.com${titleTag.attr('href')}`;

                // 날짜
                const dateEm = writeInfo.find('em').eq(2);
                let date = dateEm.text().trim().split(' ')[0];

                // 결과 추가
                result.push({ title, link, date });
            }
        });

        console.log({ "베리타스알파": result });
        return result;
    } catch (error) {
        console.error(`페이지를 가져오는 데 실패함: ${error.message}`);
        return [];
    }
}

// 약 0초
// crawlVeritas();
module.exports = crawlVeritas;
