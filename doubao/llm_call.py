import http.client
import json

def doubao_call(content):

   conn = http.client.HTTPSConnection("ark.cn-beijing.volces.com")
   payload = json.dumps({
      "model": "doubao-seed-1-6-lite-251015",
      "messages": [
         {
            "role": "system",
            "content": "You are a helpful assistant."
         },
         {
            "role": "user",
            "content": content
         }
      ]
   })
   headers = {
      'Authorization': 'Bearer dc2bd008-7c45-4744-ba12-ec6754d8c1a1',
      'Content-Type': 'application/json'
   }
   conn.request("POST", "/api/v3/chat/completions", payload, headers)
   res = conn.getresponse()
   data = res.read()
   result = data.decode("utf-8")
   result_json = json.loads(result)["choices"][0]["message"]["content"]
   return result_json

if __name__ == "__main__":
   result = doubao_call("你好")
   print(result)