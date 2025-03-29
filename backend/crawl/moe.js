const puppeteer = require('puppeteer');

async function crawlMoe() {
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  const url = 'https://www.moe.go.kr/boardCnts/listRenew.do?boardID=294&m=020402&s=moe';
  await page.goto(url, { waitUntil: 'networkidle2' });

  const result = await page.evaluate(() => {
    const rows = document.querySelectorAll('#txt > section > div:nth-child(2) > div > table > tbody > tr');
    const data = [];

    rows.forEach(row => {
      const titleTag = row.querySelector('td:nth-child(2) a');
      const dateTag = row.querySelector('td:nth-child(4)');

      if (titleTag && dateTag) {
        // 제목
        const title = titleTag.textContent.trim();
        const onclickValue = titleTag.getAttribute('onclick');

        // 자바스크립트 함수에서 값 추출
        const match = /goView\('(\d+)', '(\d+)', '(\d+)', null, '(\w)', '(\d+)', '(\w)', ''\);/.exec(onclickValue);
        if (match) {
          const [_, var1, var2, var3, var4, var5, var6] = match;
          const link = `https://www.moe.go.kr/boardCnts/viewRenew.do?boardID=${var1}&boardSeq=${var2}&lev=${var3}&searchType=null&statusYN=${var4}&page=${var5}&s=moe&m=020402&opType=${var6}`;

          // 날짜 및 결과 보내기
          data.push({ title, link, date: dateTag.textContent.trim().replace(/-/g, '.') });
        }
      }
    });

    return data;
  });

  console.log(result);

  await browser.close();
  return result;
}

// 약 2초
// crawlMoe();
module.exports = crawlMoe;
