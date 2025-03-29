const axios = require('axios');
const cheerio = require('cheerio');

async function crawlUnn() {
  try {
    const url = 'https://news.unn.net/news/articleList.html?view_type=sm';
    
    // HTTP 요청 헤더 설정 (403 방지)
    const headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
      'Referer': 'https://www.google.com/',
    };

    // 웹페이지 요청 (헤더 추가)
    const { data } = await axios.get(url, { headers });

    const $ = cheerio.load(data); // HTML 파싱
    const newsList = $('#section-list > ul > li'); // 뉴스 목록 선택

    const result = [];

    newsList.each((_, el) => {
      const titleTag = $(el).find('div h4 a');
      const writeInfo = $(el).find('.byline');

      if (titleTag.length && writeInfo.length) {
        // 제목 및 링크
        const title = titleTag.text().trim();
        const link = 'https://news.unn.net' + titleTag.attr('href');

        // 날짜
        const dateEm = writeInfo.find('em').eq(2);
        const date = dateEm.length ? dateEm.text().trim().split(' ')[0] : '';

        // 결과 추가
        result.push({ title, link, date });
      }
    });

    console.log({ "UNN": result });
    return result;
  } catch (error) {
    console.error('크롤링 실패:', error.message);
    return [];
  }
}

// 약 0초
// crawlUnn();
module.exports = crawlUnn;
