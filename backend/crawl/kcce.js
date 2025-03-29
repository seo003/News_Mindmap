const axios = require('axios');
const cheerio = require('cheerio');

// 웹사이트 URL 설정
const url = 'https://www.kcce.or.kr/web/board/1485.do';

// title_from_link 함수
async function title_from_link(link) {
  try {
    const { data } = await axios.get(link);
    const $ = cheerio.load(data);

    // 제목 추출
    const titleTag = $('#container > div.content_wrap > div.content > div.con_wrap > table > tbody > tr:nth-child(1) > th').text().trim();
    return titleTag || null;
  } catch (error) {
    console.error('Error fetching title from link:', error);
    return null;
  }
}

// 웹페이지 요청 (비동기 함수로 변경)
async function crawlKcce() {
  try {
    const { data } = await axios.get(url);
    const $ = cheerio.load(data);

    // 뉴스 데이터 선택
    const newsTable = $('#form1 > fieldset > div.list_tbl > table > tbody');
    const newses = newsTable.find('tr');

    const result = [];

    for (let i = 1; i < newses.length; i++) {  // 첫 번째 tr을 건너뛰기
      const info = $(newses[i]).find('td');
      const titleTag = info.eq(1).find('a');

      if (titleTag.length) {
        // 링크
        const linkNo = titleTag.attr('onclick').split("'")[1];
        const link = `https://www.kcce.or.kr/web/board/1485.do?mode=view&schBcode=&schCon=&schStr=&pageIndex=1&pageUnit=20&idx=${linkNo}`;

        // 제목
        let title = titleTag.text().trim();
        if (title.includes("...")) {
          title = await title_from_link(link);
        }

        // 날짜
        const dateStr = info.eq(4).text().trim();
        const date = dateStr.replace('-', '.');

        // 결과 추가
        result.push({ title, link, date });
      }
    }

    console.log(result);
    return result;
  } catch (error) {
    console.error('Error fetching main page:', error);
    return [];
  }
}

// 약 4초
// crawlKcce();
module.exports = crawlKcce;