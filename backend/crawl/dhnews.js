const axios = require('axios');
const cheerio = require('cheerio');

async function crawlDhNews() {
  try {
    const url = 'https://dhnews.co.kr/news/cate/';

    // 웹페이지 요청
    const { data } = await axios.get(url);

    // HTML 파싱
    const $ = cheerio.load(data);
    const newsList = $('div#listWrap div.listPhoto');

    const result = [];

    newsList.each((_, el) => {
      const titleTag = $(el).find('dl dt a');
      const dateTag = $(el).find('dd.winfo span.date');

      if (titleTag.length && dateTag.length) {
        const title = titleTag.text().trim();
        const date = dateTag.text().trim();
        const link = 'https://dhnews.co.kr' + titleTag.attr('href');

        result.push({ title, link, date });
      }
    });

    console.log({ "대학저널": result });
    return result;
  } catch (error) {
    console.error('크롤링 실패:', error.message);
    return [];
  }
}

// 약 1초
// crawlDhNews()
module.exports = crawlDhNews;
