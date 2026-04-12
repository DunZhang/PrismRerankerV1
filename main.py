import datasets
lang = 'zh'  # or any of the 16 languages
miracl = datasets.load_dataset('miracl/miracl', lang, trust_remote_code=True)

# training set:
for data in miracl['train']:  # or 'dev', 'testA'
    query_id = data['query_id']
    query = data['query']
    positive_passages = data['positive_passages']
    negative_passages = data['negative_passages']
    print(positive_passages[0])
    print(negative_passages[0])
    break
    # for entry in positive_passages:  # OR 'negative_passages'
    #     docid = entry['docid']
    #     title = entry['title']
    #     text = entry['text']
