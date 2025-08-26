import os
from typing import Optional, Dict, List
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 加载.env文件
load_dotenv()


class GeminiLLM:
    """
    Google Gemini LLM类，使用官方的Google GenAI SDK。
    支持Gemini 2.5 Flash模型进行文本生成。
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        """
        初始化Gemini LLM客户端。

        Args:
            api_key: Google API密钥。如果为None，将尝试从GOOGLE_API_KEY环境变量获取
            model: 要使用的模型名称。默认为gemini-2.5-flash
        """
        self.model = model

        # 从参数或环境变量获取API密钥
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key is None:
                raise ValueError("必须提供API密钥或在.env文件中设置GOOGLE_API_KEY")

        # 初始化客户端
        self.client = genai.Client(api_key=api_key)

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> str:
        """
        使用Gemini模型生成文本。

        Args:
            prompt: 输入文本提示
            system_instruction: 系统指令，用于指导模型行为
            max_tokens: 最大生成token数
            temperature: 采样温度 (0.0 到 2.0)
            stream: 是否使用流式响应

        Returns:
            生成的文本响应
        """
        # 准备生成配置
        config_params = {}
        if max_tokens:
            config_params["max_output_tokens"] = max_tokens
        if temperature is not None:
            config_params["temperature"] = temperature
        if system_instruction:
            config_params["system_instruction"] = system_instruction

        config = types.GenerateContentConfig(**config_params) if config_params else None

        try:
            if stream:
                return self._generate_stream(prompt, config)
            else:
                response = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                return response.text
        except Exception as e:
            raise Exception(f"生成文本时出错: {str(e)}")

    def _generate_stream(
        self, prompt: str, config: Optional[types.GenerateContentConfig]
    ) -> str:
        """
        生成流式响应。

        Args:
            prompt: 输入文本提示
            config: 生成配置

        Returns:
            完整的生成文本
        """
        response_text = ""
        response_stream = self.client.models.generate_content_stream(
            model=self.model, contents=prompt, config=config
        )

        for chunk in response_stream:
            response_text += chunk.text

        return response_text

    def generate_with_image(
        self, prompt: str, image_path: str, system_instruction: Optional[str] = None
    ) -> str:
        """
        使用图像输入生成文本（多模态）。

        Args:
            prompt: 文本提示
            image_path: 图像文件路径
            system_instruction: 系统指令，用于指导模型行为

        Returns:
            生成的文本响应
        """
        try:
            from PIL import Image

            image = Image.open(image_path)

            config_params = {}
            if system_instruction:
                config_params["system_instruction"] = system_instruction

            config = (
                types.GenerateContentConfig(**config_params) if config_params else None
            )

            response = self.client.models.generate_content(
                model=self.model, contents=[image, prompt], config=config
            )

            return response.text

        except ImportError:
            raise ImportError(
                "图像处理需要PIL (Pillow)库。请使用以下命令安装: pip install Pillow"
            )
        except Exception as e:
            raise Exception(f"使用图像生成文本时出错: {str(e)}")

    def chat_session(self) -> "GeminiChat":
        """
        创建一个多轮对话的聊天会话。

        Returns:
            用于对话的GeminiChat实例
        """
        return GeminiChat(self.client, self.model)


class GeminiChat:
    """
    与Gemini进行多轮对话的聊天会话类。
    """

    def __init__(self, client: genai.Client, model: str):
        """
        初始化聊天会话。

        Args:
            client: GenAI客户端实例
            model: 要使用的模型名称
        """
        self.client = client
        self.model = model
        self.chat = client.chats.create(model=model)

    def send_message(self, message: str) -> str:
        """
        在聊天会话中发送消息。

        Args:
            message: 要发送的消息

        Returns:
            模型的响应
        """
        try:
            response = self.chat.send_message(message)
            return response.text
        except Exception as e:
            raise Exception(f"发送消息时出错: {str(e)}")

    def get_history(self) -> List[Dict[str, str]]:
        """
        获取聊天历史。

        Returns:
            包含角色和内容的消息列表
        """
        history = []
        try:
            for message in self.chat.get_history():
                history.append(
                    {
                        "role": message.role,
                        "content": message.parts[0].text if message.parts else "",
                    }
                )
        except Exception as e:
            raise Exception(f"获取聊天历史时出错: {str(e)}")

        return history


# 使用示例
if __name__ == "__main__":
    # 初始化Gemini LLM
    llm = GeminiLLM()  # 将使用环境变量中的GOOGLE_API_KEY

    # 简单文本生成
    response = llm.generate_text("什么是人工智能？")
    print("响应:", response)

    # 使用系统指令
    response_with_system = llm.generate_text(
        "解释量子计算",
        system_instruction="你是一位乐于助人的物理学教授。请清晰简洁地解释概念。",
    )
    print("带系统指令的响应:", response_with_system)

    # 聊天会话示例
    chat = llm.chat_session()
    response1 = chat.send_message("你好！我有2只猫。")
    print("聊天响应1:", response1)

    response2 = chat.send_message("我家里有多少只爪子？")
    print("聊天响应2:", response2)
