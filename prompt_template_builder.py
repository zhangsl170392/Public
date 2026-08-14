import pandas as pd
from langchain_core.prompts import PromptTemplate

class PromptBuilder:
    """创建RAG prompt模板"""

    @staticmethod
    def basic_template(context: str, question: str) -> str:
        """基础问答模板（企业知识库风格）"""
        return f"""
        你是一个基于知识库的专业助手。请严格根据以下【参考内容】来回答用户的问题。
        
        如果你无法从【参考内容】中找到答案，必须明确回答"根据现有资料无法回答"，
        绝对不能使用你自己的知识来编造答案。
        
        【参考内容】
        {context}
        
        【问题】
        {question}
        
        【回答】"""

    @staticmethod
    def chat_template(context: str, question: str) -> str:
        """自然对话模板（智能客服风格）"""
        return f"""
        以下是知识库中的相关信息。
        {context}

        用户的问题是：{question}
        请你基于上面的信息，用友好、自然的语气回答用户的问题。如果信息不足，请坦诚告知。
        """

    @staticmethod
    def structured_template(context: str, question: str) -> str:
        """结构化输出模板（数据分析风格）"""
        return f"""
                根据以下【参考内容】，回答用户的问题。回答请遵循以下格式：
                一、核心结论：
                二、事实依据：引用参考内容中的原文
                三、补充说明：（如果没有则不写）

                【参考内容】
                {context}

                【问题】
                {question}
                """
    @staticmethod
    def fewshot_template(context: str, question: str) -> str:
        """少样本模板：通过示例规范模型行为，示范如何正确处理上下文与生成答案[reference:5]"""
        return f"""
                你是一个严谨的知识库问答助手。请严格遵循下方示例的格式和风格，根据【参考内容】回答问题。

                【示例1】
                【参考内容】: 企业采用混合办公模式后，员工满意度提升了15%。
                【问题】: 混合办公有什么效果？
                【回答】: 根据资料，混合办公模式使员工满意度提升了15%。

                【示例2】
                【参考内容】: 服务器维护时间为本周六凌晨2点至4点。
                【问题】: 服务器什么时候维护？
                【回答】: 根据资料，服务器将于本周六凌晨2点至4点进行维护。
                --------------------------------------

                现在，请基于下方参考内容回答问题：

                【参考内容】
                {context}

                【问题】
                {question}

                【回答】"""

basic_langchain_template = PromptTemplate(
    input_variables = ["context", "question"],
    template = PromptBuilder.basic_template(context="{context}", question="{question}")
)

def create_langchain_prompt(template_type="basic"):
    """根据类型返回 LangChain PromptTemplate 对象"""
    templates = {
        "basic": PromptBuilder.basic_template,
        "chat": PromptBuilder.chat_template,
        "structured": PromptBuilder.structured_template,
        "fewshot": PromptBuilder.fewshot_template
    }

    template_func = templates.get(template_type, templates["basic"])
    return PromptTemplate(
        input_variables = ["context", "question"],
        template = template_func(context="{context}", question="{question}")
    )

# 使用示例
if __name__ == "__main__":
    # 示例：不同模板效果对比
    sample_context = "公司2026年Q1营收为50亿元，同比增长20%。"
    sample_question = "公司Q1营收情况如何？"

    print("=== 基础模板 ===")
    print(PromptBuilder.basic_template(context=sample_context, question=sample_question))
    print("\n=== 结构化模板 ===")
    print(PromptBuilder.structured_template(context=sample_context, question=sample_question))
    print("\n=== 少样本模板 ===")
    print(PromptBuilder.fewshot_template(sample_context, sample_question))

    # 使用LangChain包装
    basic_prompt = create_langchain_prompt("basic")
    formatted_prompt = basic_prompt.format(context=sample_context, question=sample_question)
    print(f"\nLangChain 格式化后的模板：\n{formatted_prompt}")
