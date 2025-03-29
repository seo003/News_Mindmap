const fs = require('fs');
const crawlChosunEdu = require('../crawl/chosunedu');
const crawlDhNews = require('../crawl/dhnews');
const crawlIncheonNews = require('../crawl/incheon');
const crawlKcce = require('../crawl/kcce');
const crawlKyosu = require('../crawl/kyosu');
const crawlMoe = require('../crawl/moe');
const crawlUnipress = require('../crawl/unipress');
const crawlUnn = require('../crawl/unn');
const crawlUsline = require('../crawl/usline');
const crawlVeritas = require('../crawl/veritas-a');
const crawlYna = require('../crawl/yna');

async function crawlAllNews() {
    try {
      // 각 크롤러를 병렬로 실행
      const results = await Promise.all([
        crawlChosunEdu(),
        crawlDhNews(),
        crawlIncheonNews(),
        crawlKcce(),
        crawlKyosu(),
        crawlMoe(),
        crawlUnipress(),
        crawlUnn(),
        crawlUsline(),
        crawlVeritas(),
        crawlYna()
      ]);
  
      // 결과를 하나로 합치기
      const mergedResults = results.flat();
  
      // 결과를 JSON 파일로 저장
      fs.writeFileSync('news.json', JSON.stringify(mergedResults, null, 2));
  
      console.log('크롤링이 완료되었습니다. news.json 파일에 저장되었습니다.');
  
    } catch (error) {
      console.error('크롤링 중 오류 발생:', error);
    }
  }
  
  crawlAllNews();
  

crawlAllNews();
