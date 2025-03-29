const axios = require('axios');
const cheerio = require('cheerio');

async function crawlUnipress() {
  try {
    const url = 'http://www.unipress.co.kr/news/articleList.html?sc_section_code=S1N1&view_type=sm';

    // 웹페이지 요청
    const { data } = await axios.get(url);

    // HTML 파싱
    const $ = cheerio.load(data);
    const newsList = $('div.list-block');

    const result = [];

    newsList.each((_, el) => {
      const titleTag = $(el).find('div.list-titles a strong');
      const linkTag = $(el).find('div.list-titles a');
      const dateTag = $(el).find('div.list-dated');

      if (titleTag.length && linkTag.length && dateTag.length) {
        // 제목 및 날짜
        const title = titleTag.text().trim();
        const link = 'http://www.unipress.co.kr' + linkTag.attr('href');

        // 날짜 추출
        let date = null;
        const match = dateTag.text().trim().match(/(\d{4})-(\d{2})-(\d{2})/);
        if (match) {
          date = `${match[1]}.${match[2]}.${match[3]}`;
        }

        result.push({ title, link, date });
      }
    });

    console.log({ "대학지성IN&OUT": result });
    return result;
  } catch (error) {
    console.error('크롤링 실패:', error.message);
    return [];
  }
}

// 약 0초
// crawlUnipress();
module.exports = crawlUnipress;