import logging
from typing import Optional

from langchain.llms.base import BaseLLM

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import ActionResult, action
from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.llm.params import llm_params
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.llm.types import Task
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.utils import new_event_dict


log = logging.getLogger(__name__)

@action(name="self_check_input",is_system_action=True)
async def self_check_input(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    llm: Optional[BaseLLM] = None,
    config: Optional[RailsConfig] = None,
):
    """Checks the input from the user.

    Prompt the LLM, using the `check_input` task prompt, to determine if the input
    from the user should be allowed or not.

    Returns:
        True if the input should be allowed, False otherwise.
    """

    user_input = context.get("user_message")

    if user_input:
        prompt = llm_task_manager.render_task_prompt(
            task=Task.SELF_CHECK_INPUT,
            context={
                "user_input": user_input,
            },
        )
        stop = llm_task_manager.get_stop_tokens(task=Task.SELF_CHECK_INPUT)

        # Initialize the LLMCallInfo object
        llm_call_info_var.set(LLMCallInfo(task=Task.SELF_CHECK_INPUT.value))

        with llm_params(llm, temperature=config.lowest_temperature):
            check = await llm_call(llm, prompt, stop=stop)

        check = check.lower().strip()
        log.info(f"Input self-checking result is: `{check}`.")

        if "yes" in check:
            return ActionResult(
                return_value=False,
                events=[
                    new_event_dict(
                        "mask_prev_user_message", intent="unanswerable message"
                    )
                ],
            )

    return True
