import json
import threading
from crawl.chosun import get_data as chosun
from crawl.dhnews import get_data as dhnews
from crawl.incheon import get_data as incheon
from crawl.kcce import get_data as kcce
from crawl.kyosu import get_data as kyosu
from crawl.moe import get_data as moe
from crawl.unipress import get_data as unipress
from crawl.unn import get_data as unn
from crawl.usline import get_data as usline
from crawl.veritas import get_data as veritas
from crawl.yna import get_data as yna
from crawl.incheonilbo import get_data as incheonilbo
from crawl.kyeongin import get_data as kyeongin


lock = threading.Lock()

def fetch_news(source, result_list):
    data = source()
    if isinstance(data, list) and data != ['error']:
        with lock:
            result_list.extend(data)

def crawl_all():
    data = []
    threads = []

    sources = [chosun, dhnews, incheon, kcce, kyosu, moe, unipress, unn, usline, veritas, yna, incheonilbo, kyeongin]

    for source in sources:
        thread = threading.Thread(target=fetch_news, args=(source, data))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # JSON 파일로 저장
    # with open("data/news.json", "w", encoding="utf-8") as f:
    #     json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"크롤링 완료: 총 {len(data)}개 기사 수집")
    return data

# crawl_all()
