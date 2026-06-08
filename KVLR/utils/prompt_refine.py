"""Offline prompt-refinement stubs for anonymous supplementary release.

The paper experiments use fixed prompts from KASA annotations. To keep the
supplement self-contained and anonymous, no external LLM service is invoked here.
"""


def refine_prompt(prompt: str, retry_times: int = 3, type: str = "t2v", image_path: str = None):
    return prompt


def refine_prompts(prompts: list[str], retry_times: int = 3, type: str = "t2v", image_paths: list[str] = None):
    return list(prompts)
