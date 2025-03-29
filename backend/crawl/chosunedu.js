const axios = require('axios');

async function crawlChosunEdu() {
  try {
    // AJAX 요청 URL
    const url = 'https://edu.chosun.com/svc/app/edu_list.php';

    // 요청 파라미터 설정
    const params = {
      catid: '14',
      pn: 1,
      rows: 10
    };

    // AJAX 요청 보내기 (헤더 설정 없이)
    const { data } = await axios.get(url, { params });

    const result = data.CONTENT.map(content => ({
      title: content.TITLE,
      link: 'https:' + content.ART_HREF,
      date: content.DATE.slice(0, 10)
    }));

    console.log({ "조선에듀": result });
    return result;

  } catch (error) {
    console.error('크롤링 실패:', error.message);
    return [];
  }
}

// 약 0초
// crawlChosunEdu();
module.exports = crawlChosunEdu;
