from langchain.chains import LLMRequestsChain, LLMChain
from langchain.prompts import PromptTemplate

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class LMRequestsChain:
    def __init__(self,llm,max_url_num = 2):
        template = """Between >>> and <<< are the raw search result text from google.
                Extract the answer to the question '{query}' or say "not found" if the information is not contained.
                Use the format
                Extracted:<answer or "not found">
                >>> {requests_result} <<<
                Extracted:"""
        PROMPT = PromptTemplate(
            input_variables=["query", "requests_result"],
            template=template,
        )
        self.chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=PROMPT))
        self.max_url_num = max_url_num

        query_prompt = PromptTemplate(
            input_variables=["query","responses"],
            template = "作为一名专业的信息总结员，我需要查询的信息为{query},根据提供的信息{responses}回答一下查询的结果。")
        self.query_chain = LLMChain(llm=llm, prompt=query_prompt)

    def __call__(self, query,target_site = ""):
        url_list = self.get_urls(query,target_site = target_site)
        print(f"查找到{len(url_list)}条url内容，现在开始解析其中的{self.max_url_num}条内容。")
        responses = []
        for url in tqdm(url_list[:self.max_url_num]):
            inputs = {
                "query": query,
                "url": url
            }

            response = self.chain(inputs)
            output = response["output"]
            responses.append(output)
        if len(responses) != 0:
            output = self.query_chain.run({"query":query,"responses":responses})
            return output
        else:
            return "查找内容为空，请更换查找词"

    def query_form_url(self,query = "LMchain是什么？",url = ""):
        assert url != "",print("url link must be set")
        inputs = {
            "query": query,
            "url": url
        }
        response = self.chain(inputs)
        return response

    def get_urls(self,query='lmchain是什么?', target_site=""):
        def bing_search(query, count=30):
            url = f'https://cn.bing.com/search?q={query}'
            headers = {
                'User-Agent': 'Mozilla/6.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                html = response.text
                # 使用BeautifulSoup解析HTML

                soup = BeautifulSoup(html, 'html.parser')
                results = soup.find_all('li', class_='b_algo')
                return [result.find('a').text for result in results[:count]]
            else:
                print(f'请求失败，状态码：{response.status_code}')
                return []
        results = bing_search(query)
        if len(results) == 0:
            return None
        url_list = []
        if target_site != "":
            for i, result in enumerate(results):
                if "https" in result and target_site in result:
                    url = "https://" + result.split("https://")[1]
                    url_list.append(url)
        else:
            for i, result in enumerate(results):
                if "https" in result:
                    url = "https://" + result.split("https://")[1]
                    url_list.append(url)
        if len(url_list) > 0:
            return url_list
        else:
            # 这里是确保在知乎里面找不到对应的内容，有相应的内容返回
            for i, result in enumerate(results):
                if "https" in result:
                    url = "https://" + result.split("https://")[1]
                    url_list.append(url)
            return url_list


