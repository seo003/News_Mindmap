const puppeteer = require('puppeteer');

async function crawlYna() {
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  await page.goto('https://www.yna.co.kr/news?site=navi_latest_depth01', {
    waitUntil: 'networkidle2',
  });

  // 페이지 로드 후 기사 데이터 가져오기
  const result = await page.evaluate(() => {
    const newsList = document.querySelectorAll(
      '#container > div.container521 > div.content03 > div.section01 > section > div > ul > li'
    );
    const currentYear = new Date().getFullYear();
    const extractedData = [];

    newsList.forEach((news) => {
      const titleTag = news.querySelector('div > div > strong > a > span');
      const dateTag = news.querySelector('div > div > span');
      const linkTag = news.querySelector('div > div > strong > a');

      if (titleTag && dateTag && linkTag) {
        // 제목 
        const title = titleTag.innerText.trim();
        // 날짜
        const dateText = dateTag.innerText.trim().split(' ')[0]; // "03-11 15:30" -> "03-11"
        const date = `${currentYear}.${dateText.replace('-', '.')}`; // "2024.03.11"
        // 링크
        const link = linkTag.href.startsWith('http') ? linkTag.href : `https://www.yna.co.kr${linkTag.getAttribute('href')}`;

        extractedData.push({ title, link, date });
      }
    });

    return extractedData;
  });

  await browser.close();

  console.log({ '연합뉴스': result });
  return result;
}

// 약 4초
// crawlYna();
module.exports = crawlYna;
