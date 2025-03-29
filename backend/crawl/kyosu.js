const axios = require('axios');
const cheerio = require('cheerio');

async function crawlKyosu() {
  try {
    const url = 'https://www.kyosu.net/news/articleList.html?view_type=sm';

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
        // 제목 및 링크크
        const title = titleTag.text().trim();
        const link = 'https://www.kyosu.net' + linkTag.attr('href');

        // 날짜 추출 
        let date = null;
        const match = dateTag.text().trim().match(/(\d{4})-(\d{2})-(\d{2})/);
        if (match) {
          date = `${match[1]}.${match[2]}.${match[3]}`;
        }

        result.push({ title, link, date });
      }
    });

    console.log({ "교수신문": result });
    return result;
  } catch (error) {
    console.error('크롤링 실패:', error.message);
    return [];
  }
}

// 약 1초
// crawlKyosu();
module.exports = crawlKyosu;
