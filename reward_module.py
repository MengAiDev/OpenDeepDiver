def calculate_reward(responses, prompts, train_dataset):
    rewards = []
    for response, prompt in zip(responses, prompts):
        answer = next((item["answer"] for item in train_dataset.data if item["prompt"] == prompt), "")
        if answer and answer in response:
            reward = 1.0
        else:
            reward = 0.0
            
        search_count = response.count("搜索：")
        if search_count > 0:
            reward += min(search_count * 0.2, 1.0)
            
        reflection_count = response.count("反思：")
        reward += min(reflection_count * 0.3, 1.5)
        
        difficulty = next((item["difficulty"] for item in train_dataset.data if item["prompt"] == prompt), "medium")
        if difficulty == "hard":
            reward += 0.5
        elif difficulty == "outlier":
            reward += 1.0
            
        if len(response) > 500:
            reward -= 0.3
            
        rewards.append(reward)
        
    return rewards