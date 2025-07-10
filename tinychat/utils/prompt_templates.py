from typing import List
from tinychat.utils.constants import (
    LLAVA_DEFAULT_IMAGE_TOKEN,
    LLAVA_DEFAULT_IMAGE_PATCH_TOKEN,
)


def get_image_token(model, model_name):
    return LLAVA_DEFAULT_IMAGE_TOKEN + "\\n "
    # if "llava" in model_name.lower():
    #     return LLAVA_DEFAULT_IMAGE_TOKEN + "\\n "
    # elif "vila" in model_name.lower():
    #     vision_config = model.get_vision_tower().vision_tower.config
    #     image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
    #     if (
    #         "downsample" in model.config.mm_projector_type
    #         or "ds" in model.config.mm_projector_type
    #     ):
    #         image_token_len = image_token_len // 4
    #     if "p32" in model_name:  # extra leading patches
    #         image_token_len += 32
    #     elif "se" in model.config.mm_projector_type:
    #         image_token_len += 2
    #     return LLAVA_DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + "\\n "
    # return ""


class BasePrompter:
    def __init__(
        self,
        system_inst,
        role1,
        role2,
        sen_spliter="\n",
        qa_spliter="\n",
        decorator: List[str] = None,
        colon=":",
    ):
        self.system_inst = system_inst  # System Instruction
        self.role1 = role1  # The name of USER
        self.role2 = role2  # The name of AI-Assistant
        self.sen_spliter = sen_spliter  # How to split system/user/assistant outputs
        self.qa_spliter = qa_spliter  # How to split Q&A rounds
        self.decorator = decorator
        self.colon = colon
        if self.decorator == None:
            self.starter = ""
            self.stopper = ""
        else:
            self.starter = self.decorator[0]
            self.stopper = self.decorator[1]
        if self.system_inst == None:
            self.template = (
                self.starter
                + self.role1
                + self.colon
                + " {prompt}"
                + self.stopper
                + self.sen_spliter
                + self.starter
                + self.role2
                + self.colon
            )
        else:

            self.template = (
                self.starter
                + self.system_inst
                + self.stopper
                + self.sen_spliter
                + self.starter
                + self.role1
                + self.colon
                + " {prompt}"
                + self.stopper
                + self.sen_spliter
                + self.starter
                + self.role2
                + self.colon
            )
        self.model_input = None

    def insert_prompt(self, input_prompt):
        self.model_input = self.template.format(prompt=input_prompt)

    def update_template(self, outputs, chunk_prefilling=0):
        if chunk_prefilling:
            self.template = (
                self.role1
                + self.colon
                + "{prompt}"
                + self.stopper
                + self.sen_spliter  # blank space
                + self.starter
                + self.role2
                + self.colon
            )
        else:
            self.template = (
                self.model_input
                + " "
                + outputs.strip()
                + self.stopper
                + self.qa_spliter
                + self.starter
                + self.role1
                + self.colon
                + "{prompt}"
                + self.stopper
                + self.sen_spliter
                + self.starter
                + self.role2
                + self.colon
            )
        self.model_input = None


class OneShotBasePrompter(BasePrompter):
    def __init__(
        self,
        oneshot_example: List[str],  # User prompt + Assistant responce
        system_inst,
        role1,
        role2,
        sen_spliter="\n",
        qa_spliter="\n",
        decorator: List[str] = None,
    ):
        super().__init__(system_inst, role1, role2, sen_spliter, qa_spliter)
        assert len(oneshot_example) == 2, "One-shot example must be a List of 2 strs."
        self.user_example = oneshot_example[0]
        self.assistant_example = oneshot_example[1]
        self.insert_prompt(self.user_example)
        self.update_template(self.assistant_example)


class EmptyPrompter(BasePrompter):
    def __init__(self):
        system_inst = ""
        role1 = ""
        role2 = ""
        sen_spliter = ""
        qa_spliter = "</s>"
        super().__init__(system_inst, role1, role2, sen_spliter, qa_spliter)


class VicunaPrompter(BasePrompter):
    def __init__(self):
        system_inst = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        role1 = "USER"
        role2 = "ASSISTANT"
        sen_spliter = " "
        qa_spliter = "</s>"
        super().__init__(system_inst, role1, role2, sen_spliter, qa_spliter)


class Llama2Prompter(OneShotBasePrompter):
    def __init__(self, short_prompt=False):
        system_inst = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        role1 = "### Human"
        role2 = "### Assistant"
        sen_spliter = "\n"
        qa_spliter = "</s>"
        user_example = "Got any creative ideas for a 10 year old's birthday?"
        if short_prompt:
            assistant_example = (
                "Of course! Here are some creative ideas for a 10-year-old's birthday party:\n"
                + "1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.\n"
                + "2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.\n"
                + "Remember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!"
            )
        else:
            assistant_example = (
                "Of course! Here are some creative ideas for a 10-year-old's birthday party:\n"
                + "1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.\n"
                + "2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.\n"
                + "3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.\n"
                + "4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.\n"
                + "5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants.\n"
                + "6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen.\n"
                + "7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges.\n"
                + "8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors.\n"
                + "Remember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!"
            )
        oneshot_example = [user_example, assistant_example]
        super().__init__(
            oneshot_example, system_inst, role1, role2, sen_spliter, qa_spliter
        )


class Llama3Prompter(BasePrompter):
    """
    Example:
    <|start_header_id|>user<|end_header_id|>

    Show me some attractions in Boston.<|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>

    """

    def __init__(self):
        system_inst = ""
        role1 = "<|start_header_id|>user<|end_header_id|>\n\n"
        role2 = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        sen_spliter = "<|eot_id|>"
        qa_spliter = ""
        colon = ""
        super().__init__(
            system_inst, role1, role2, sen_spliter, qa_spliter, colon=colon
        )


class QwenPrompter(BasePrompter):
    def __init__(self):
        system_inst = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        role1 = "<|im_start|>user\n"
        role2 = "<|im_start|>assistant\n"
        sen_spliter = "<|im_end|>\n"
        qa_spliter = ""
        colon = ""
        super().__init__(
            system_inst, role1, role2, sen_spliter, qa_spliter, colon=colon
        )


class LlavaLlamaPrompter(BasePrompter):
    def __init__(self):
        system_inst = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        role1 = "USER"
        role2 = "ASSISTANT"
        sen_spliter = " "
        qa_spliter = "</s>"
        super().__init__(system_inst, role1, role2, sen_spliter, qa_spliter)


class LlavaLlama3Prompter(BasePrompter):
    """
    Example:
    <|start_header_id|>user<|end_header_id|>

    Show me some attractions in Boston.<|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>

    """

    def __init__(self):
        system_inst = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. "
            + "You are able to understand the visual content that the user provides, "
            + "and assist the user with a variety of tasks using natural language."
        )
        role1 = "<|start_header_id|>user<|end_header_id|>\n\n"
        role2 = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        sen_spliter = "<|end_of_text|>"
        qa_spliter = ""
        colon = ""
        super().__init__(
            system_inst, role1, role2, sen_spliter, qa_spliter, colon=colon
        )


class FalconSimplePrompter(BasePrompter):
    def __init__(self):
        system_inst = None
        role1 = "User"
        role2 = "Assistant"
        sen_spliter = "\n\n"
        qa_spliter = "\n\n"
        super().__init__(system_inst, role1, role2, sen_spliter, qa_spliter)


class FalconPrompter(BasePrompter):
    def __init__(self):
        system_inst = (
            "The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Falcon, "
            + "and a human user, called User. In the following interactions, User and Falcon will converse in natural language, "
            + "and Falcon will answer User's questions. Falcon was built to be respectful, polite and inclusive. "
            + "Falcon was built by the Technology Innovation Institute in Abu Dhabi. "
            + "Falcon will never decline to answer a question, and always attempts to give an answer that User would be satisfied with. "
            + "It knows a lot, and always tells the truth. The conversation begins."
        )
        role1 = "User"
        role2 = "Assistant"
        sen_spliter = "\n"
        qa_spliter = "\n"
        super().__init__(system_inst, role1, role2, sen_spliter, qa_spliter)


class MPTPrompter(BasePrompter):
    def __init__(self):
        system_inst = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        role1 = "### Human"
        role2 = "### Assistant"
        sen_spliter = "\n"
        qa_spliter = "\n"
        super().__init__(system_inst, role1, role2, sen_spliter, qa_spliter)


class MPTChatPrompter(BasePrompter):
    def __init__(self):
        system_inst = (
            "system\n"
            + "- You are a helpful assistant chatbot trained by MosaicML.\n"
            + "- You answer questions.\n"
            + "- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.\n"
            + "- You are more than just an information source, you are also able to write poetry, short stories, and make jokes."
        )
        role1 = "user"
        role2 = "assistant"
        sen_spliter = "\n"
        qa_spliter = "\n"
        decorator = ["<|im_start|>", "<|im_end|>"]
        super().__init__(system_inst, role1, role2, sen_spliter, qa_spliter, decorator)


class NVILAPrompter(BasePrompter):
    def __init__(self):
        system_inst = "system\n" + "You are a helpful assistant<|im_end|>\n"
        role1 = "user"
        role2 = "assistant"
        sen_spliter = "\n"
        qa_spliter = "\n"
        decorator = ["<|im_start|>", "<|im_end|>"]
        super().__init__(
            system_inst, role1, role2, sen_spliter, qa_spliter, decorator, "\n"
        )

class InternVL3Prompter(BasePrompter):
    def __init__(self):
        system_inst = "system\n你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。你可以理解用户提供的视觉内容，并使用自然语言帮助用户完成各种任务。"
        role1 = "user"
        role2 = "assistant"
        sen_splitter = "\n"
        qa_splitter = "\n"
        decorator = ["<|im_start|>", "<|im_end|>"]
        super().__init__(
            system_inst, role1, role2, sen_splitter, qa_splitter, decorator, "\n"
        )


def get_prompter(model_type, model_path="", short_prompt=False, empty_prompt=False):
    if empty_prompt:
        return EmptyPrompter()
    if model_type.lower() == "llama":
        if "vicuna" in model_path.lower():
            return VicunaPrompter()
        elif (
            "llama-3" in model_path.lower() or "llama3" in model_path.lower()
        ) and "30b" not in model_path.lower():
            if "vila" in model_path.lower():
                # with system prompt by default
                return LlavaLlama3Prompter()
            else:
                return Llama3Prompter()
        elif "llava" in model_path.lower() or "vila" in model_path.lower():
            return LlavaLlamaPrompter()
        else:
            return Llama2Prompter(short_prompt)
    elif model_type.lower() == "falcon":
        # return FalconPrompter()
        return FalconSimplePrompter()
    elif "qwen" in model_path.lower() or "qwen" in model_type.lower():
        return QwenPrompter()
    elif model_type.lower() == "mpt":
        if "mpt" and "chat" in model_path.lower():
            return MPTChatPrompter()
        else:
            return MPTPrompter()
    elif model_type.lower() == "nvila":
        return NVILAPrompter()
    elif model_type.lower() == "internvl3":
        return InternVL3Prompter()
    else:
        raise ValueError(f"model type {model_type} is not supported")


def get_stop_token_ids(model_type, model_path=""):
    if model_type.lower() == "llama":
        if (
            "llama-3" in model_path.lower() or "llama3" in model_path.lower()
        ) and "30b" not in model_path.lower():
            # llama3
            return [128001, 128009]
        return []
    elif model_type.lower() == "qwen" or model_type.lower() == "internvl3":
        return [151643, 151645]
    elif model_type.lower() == "falcon":
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    elif model_type.lower() == "mpt":
        if "mpt" and "chat" in model_path:
            return [50278, 0]
        else:
            return []
    elif model_type.lower() == "nvila":
        return [151645]
    else:
        raise ValueError(f"model type {model_type} is not supported")
