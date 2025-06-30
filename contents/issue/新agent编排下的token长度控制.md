# 一、问题：调整了agent编排过程后的计算资源消耗

## 问题摘要

- API与本地部署的调试问题
- 输入/输出 token数量的控制问题
  - qwen的proposal比较简短，llama的proposal比较长
- prompt输入的合理性
  - 因为是基于原来的输入进行的修改，暂时有点粗暴
  - 比如**后两个**step中citation的输入

## 计算的相关参数设置

- GAG**原始**的设置
  - 每篇文章的计算时间消耗：1个`step`（一般是2个迭代）
  - 每轮生成50篇文章，最大文章数量：1200
- **更改编排后**的设置：
  - 每篇文章的计算时间消耗：==3个==`step`（proposal、related work、paper（title、keywords、abstract））
- 1个`step`包含：**最多3个迭代**
  逻辑稳定后，生成一篇文章需要**2个迭代**
  如需了解，代码见附录
  - thought：**调用大模型思考**
  - action：如果需要调用工具，则进入该处理逻辑
    - 如果不需要调用工具，解析返回值时会解析出一个finish，退出迭代



## API调试

- API调试
  - 为什么只生成了3篇文章，且有一篇是空的
    - 检查服务器日志
    - 初步调查：超时（API限制）会导致文章为空，存在一定的不可控因素
- **GAG原来的实现** 和 **agentscope本身的框架**没有做任何的针对429 rate limit的处理，一般直接返回为空——生成的abstract为空，不加入数据库中
  - 原来的逻辑对生成的质量可能就有影响
  - 在**新的编排逻辑**下，有可能在完成前就有一个问题，会导致计算资源的浪费：完成了前面的step，**最终的step没有办法完成**
  - 单次测试：每轮任务量为生成10篇——最终**第一轮成功生成6篇**
    - 成功生成的文章index：（07, 08, 11, 12, 13, 15），原本要生成的文章index范围：07-16



## 本地部署方式

- 本地部署调试
  - 在group agent在调用write proposal过程中超时
    - 原因：每轮50个group进行该流程时太多

- 存在单轮模拟的集体超时
  - 暂时在寻找最佳的每轮任务量（生成的文章数量）以及RPC通信 timeout设置，这些是没有办法在API调试方式下找到的
- 由于本地部署的模型为llama3-70B，API使用的模型为qwen2-72B，有些问题（如对模型生成内容的解析）也==没有办法在API调试的环境下复现==
  - 需要优化的地方：且由于llama3-70B性能要优于qwen2-72B，有些时候会导致轮到`step 3`生成paper时，输入的token量较多，可能会超出模型的限制
    - **需要优化输入的token量，以及前面过程输出的token量**



## 输入与输出

### case

#### 成功的生成

qwen

```json
{
    "id": "6085baf1f298426c99254c28b8440c5d",
    "timestamp": "2025-04-21 16:53:59",
    "name": "group_6",
    "content": {
        "topic": "IR",
        "keywords": [
            "Bayesian Inference",
            "Machine Learning",
            "Information Retrieval",
            "Personalized IR"
        ],
        "abstract": "This paper proposes a novel approach to improving Information Retrieval (IR) systems by integrating Bayesian Inference and Machine Learning techniques. The aim is to develop a personalized and adaptive IR system capable of understanding user intent and context, thereby enhancing search accuracy and relevance. By dynamically adjusting retrieval factors based on user interactions, the system offers a more sophisticated search experience.",
        "citations": [
            "Probabilistic Latent Semantic Analysis",
            "Signal Detection Using ICA: Application to Chat Room Topic Spotting"
        ],
        "title": "Enhancing Information Retrieval through Bayesian Inference and Machine Learning Integration",
        "proposal": "The proposed research aims to investigate the integration of Bayesian Inference and Machine Learning techniques to enhance Information Retrieval (IR) systems. The goal is to develop a personalized and adaptive IR system that can better understand user intent and context, thereby providing higher-quality search results. The system will dynamically adjust retrieval factors based on user interactions and feedback, leading to improved relevance and accuracy of search results.",
        "related_work": "The proposed research on integrating Bayesian Inference and Machine Learning techniques for enhancing Information Retrieval (IR) systems is closely related to the work of Klaus P. Jantke [14], who explored the use of these methods to improve the relevance and accuracy of search results. Similarly, Thomas Kolenda [10] applied Independent Component Analysis (ICA) for signal detection in chat room topic spotting, demonstrating the effectiveness of statistical methods in understanding textual data. These works highlight the importance of statistical and probabilistic approaches in IR and provide valuable insights into the development of our personalized and adaptive IR system.",
        "searched_items": "",
        "author_ids": [
            "65",
            "68",
            "66",
            "67",
            "64"
        ],
        "success": "True",
        "searched_keywords": [

        ],
        "cited_num": 2,
        "refine_round": 1
    },
    "role": "assistant"
}
```

llama



#### 失败的生成

qwen，倒在了最后一步（keywords、abstract、title）

```json
{
    "id": "2adaf89f8e214998bc28b1f014185891",
    "timestamp": "2025-04-21 16:54:01",
    "name": "group_3",
    "content": {
        "topic": "ML",
        "keywords": [

        ],
        "abstract": "",
        "citations": [
            "Stochastic Attribute Selection Committees",
            "Boosting the Margin: A New Explanation for the Effectiveness of Voting Methods",
            "An Empirical Comparison of Decision Trees and Other Classification Methods"
        ],
        "title": "",
        "proposal": "The proposed research aims to investigate the effectiveness of integrating ensemble methods, specifically focusing on Boosting and Stochastic Attribute Selection Committees (SASC), to improve the performance of decision tree learning algorithms. By combining these techniques, we expect to enhance the accuracy and robustness of the classification models while maintaining computational efficiency. The research will involve conducting empirical evaluations on diverse datasets to compare the proposed integrated approach against standalone Boosting, SASC, and other popular classification methods. Additionally, we will explore the impact of the integrated approach on feature selection and the interpretability of the decision tree models.",
        "related_work": "The proposed research aims to investigate the effectiveness of integrating ensemble methods, specifically focusing on Boosting and Stochastic Attribute Selection Committees (SASC), to improve the performance of decision tree learning algorithms. By combining these techniques, we expect to enhance the accuracy and robustness of the classification models while maintaining computational efficiency. The research will involve conducting empirical evaluations on diverse datasets to compare the proposed integrated approach against standalone Boosting, SASC, and other popular classification methods. Additionally, we will explore the impact of the integrated approach on feature selection and the interpretability of the decision tree models. Recent studies have shown that ensemble methods can significantly improve the classification performance of decision tree learners (Zheng, 1998). Specifically, Boosting has been shown to effectively increase the margin of classification models, leading to better generalization (Schapire, 1997). Moreover, the combination of Boosting and SASC has been demonstrated to further improve the performance of decision tree learning techniques (Zheng, 1998).",
        "searched_items": "",
        "author_ids": [
            "2705",
            "359",
            "357",
            "2062",
            "2061"
        ],
        "success": "True",
        "searched_keywords": [

        ],
        "cited_num": 3,
        "refine_round": 1
    },
    "role": "assistant"
}
```

### I/O proposal：

- 输入
  - `role_description`
  - `researcher`
  - `past_context`——最近的讨论记忆
  - `write_memory`——曾经的写作记忆
  - `research_topic`
  - `searched_info`
- 输出
  - `proposal`
  - `citations`

### I/O related work：

- 输入
  - `role_description`
  - `researcher`
  - `past_context`——最近的讨论记忆
  - `write_memory`——曾经的写作记忆
  - **前一个step的输出**：`topic`、`proposal`、`citations`
  - `searched_info`
- 输出
  - `related_work`
  - `citations`

### I/O paper（title、keywords、abstract）：

- 输入
  - `role_description`
  - `researcher`
  - `past_context`——最近的讨论记忆
  - `write_memory`——曾经的写作记忆
  - **前一个step的输出**：`topic`、`proposal`、`related_work`、`citations`
  - `searched_info`
- 输出
  - `title`
  - ` keywords`
  - `abstract`
  - `citations`

# 二、已解决的问题

- 上周五提到的，工具的调用问题
  - 通过**调整prompt的内容**已经解决
  - 过程分析：在写paper abstract阶段，大模型将(1)理解为了(2)，该阶段原本不需要调用工具
    - (1)：不要调用工具来检索
    - (2)：调用除了检索工具之外的其它工具
  - 导致调用了一些不存在的工具（幻觉）



# 三、代码/Prompt附录

## 每次`step`的过程

```python
def step(self,
            agent_msgs: List[Msg]= [],
            use_tools:bool = False,
            return_tool_exec_only: bool = False,
            return_intermediate_steps :bool = False,
            ) -> Msg:
    if not isinstance(agent_msgs,list):
        agent_msgs = [agent_msgs]
    intermediate_steps = []
    steps_idx = 0
    
    prompt = []
    ### 生成并拼接工具的prompt
    if use_tools:
        res_tool_msg = self.call_manager_func(
            "get_prompt_tool_msgs")
        

    while(steps_idx < self.max_retrys):
        
        memory_msgs = self.get_agent_memory_msgs().content
        if use_tools and steps_idx < self.max_tool_iters:
            if return_tool_exec_only:
                prompt = [*agent_msgs, 
                        #   *memory_msgs,
                            *res_tool_msg.content]
            else:
                prompt = [*agent_msgs, 
                            *memory_msgs,
                            *res_tool_msg.content]
        else:
            prompt = [*agent_msgs,
                        *memory_msgs]
        
        steps_idx +=1
        ### Step 1: Thought
        # Generate LLM response
        
        msg_finish = self.call_agent_reply_prompt(prompt)	### 在这里调用一次大模型进行thought
        msg_finish.content # 强制阻塞

        if msg_finish.get("fail",False):
            print(msg_finish.content)
            continue

        if msg_finish.get("finish",False):
            self.agent.speak(f" ITER {steps_idx}, STEP: FINISH".center(70, "#"))
            if return_intermediate_steps:
                msg_finish.update({"intermediate_steps":intermediate_steps})
            # Skip the next steps if no need to call tools
            self.call_agent_func("clear_short_memory").content
            return msg_finish
        
        ### Step 2: Action
        self.agent.speak(f" ITER {steps_idx}, STEP: ACTION ".center(70, "#"))
        try:
            execute_results = self.call_manager_func(msg_finish.func,
                                                    kwargs={
                                                    "function_call_msg":msg_finish.content}).content
        except Exception as e:
            execute_results = []
            print(msg_finish, e)
        assert isinstance(execute_results,list)
        intermediate_steps.extend(execute_results)

        # Prepare prompt for execution results
        execute_results_prompt = "\n".join(
            [
                FUNCTION_RESULT_PROMPT.format_map(res_one[1])
                for res_one in execute_results
            ],
        )
        # Add title
        execute_results_prompt = (
            FUNCTION_RESULT_TITLE_PROMPT + execute_results_prompt
        )

        # Note: Observing the execution results and generate response are
        # finished in the next loop. We just put the execution results
        # into memory, and wait for the next loop to generate response.

        # Record execution results into memory as a message from the system
        msg_res = Msg(
            name = self.agent.name,
            content=execute_results_prompt,
            role="assistant",
        )
        self.agent.speak(msg_res)
        self.agent.observe([msg_finish,msg_res])
        
        if return_tool_exec_only and \
            steps_idx == self.max_tool_iters:
            if return_intermediate_steps:
                msg_finish.update({"intermediate_steps":intermediate_steps})
            self.call_agent_func("clear_short_memory").content
            return msg_finish
    
    msg_finish = Msg(
        "system",
        "The agent has reached the maximum iterations.",
        role="system",
    )

    if return_intermediate_steps:
        msg_finish.update({"intermediate_steps":intermediate_steps})

    self.call_agent_func("clear_short_memory").content
    return msg_finish
```

## Proposal Prompt

```yaml
write_proposal: &write_proposal |-
  {role_description}

  You have discussed your research with other researchers. Other researchers include:
  {researcher}

  Your discussion are listed as follows: 
  {past_context}

  The version of your proposal now:
  {current_proposal}

  Now you need to **propose a academic proposal** based on your reasearch topic and searched information.

  The searched information:
  {searched_info}

  {write_memory}


  - If you want to generate a version of proposal, your proposal should cite 2 to 5 papers, 
  you can decide the number according to your needs. 
  And you should respond in this json format, which can be loaded by json.loads:

  proposal: (The proposal you think)
  citations: (List; The list of the paper names you want to cite. This should include all the titles of the papers you cite. You should include the papers you have searched.)

  Respond only the proposal or action at one time!
  
  Now respond:
  {agent_scratchpad}
```

## Related Work Prompt

- 部分参数解析
  - `{current_related_work}`：

```yaml
write_related_work: &write_related_work |-
  {role_description}

  You have discussed your research with other researchers. Other researchers include:
  {researcher}

  Your discussions are listed as follows:
  {past_context}

  The current version of your related work section is:
  {current_related_work}

  Now you need to write or improve the **related work section** for your academic paper based on your research proposal and the searched information.

  The searched information:
  {searched_info}

  {write_memory}


  - When generating a version of the related work section, ensure that it cites {min_citations} to {max_citations} papers. You may choose the precise number based on your needs.
  - Your output must strictly follow the JSON format below (which must be loadable by json.loads):

  "related_work": (This is the related work section you propose),
  "citations": (List; The list of the paper names you want to cite. This should include all the titles of the papers you cite. You should include the papers you have searched.)

  Respond only with the related work section or action at one time!

  Now respond:
  {agent_scratchpad}
```

## abstract prompt

```yaml
write_article: &write_article |-
  {role_description}

  You have discussed your research with other researchers. Other researchers include:
  {researcher}

  Your discussion are listed as follows: 
  {past_context}

  Based on your proposal, related work and other information, finish or refine your paper.
  **Do NOT call or invoke any tools or functions.**  Just output the completed paper directly in JSON format.

  A papar should include the following attributes: 
    title: The title should be concise yet descriptive, providing a clear indication of the paper's topic and scope. This can be different from your topic, It is relatively accurate and clear.
    keywords: These are specific terms or phrases that encapsulate the core topics of your paper. Keywords make your paper searchable within academic databases and help readers quickly understand the paper's focus areas.
    abstract: The abstract is a brief summary of your research paper. It should provide an overview of the research question, methodology, results, and conclusions. 
    citations: A list of the paper names you want to cite, source from proposal and related section.


  The version of your paper now: 
  {current_paper}

  {searched_info}

  {write_memory}

        
  - If you want to generate a version of paper, your paper should cite {min_citations} to {max_citations} papers based on your proposal and related work section, 
  you can decide the number according to your needs. 
  And you should respond in this json format, which can be loaded by json.loads:
  
  title: (The title of your paper, be concise and specific)
  keywords: (The keywords for your next paper)
  abstract: (The topic content of the paper you want to write)
  citations: (List; The list of the paper names you want to cite. This should include all the titles of the papers you cite. You should include the papers you have searched.)
    
  Respond **only** with the JSON object—do not include any additional wrappers, logs, or tool‑call metadata.
         
  Now respond:
  {agent_scratchpad}
```

