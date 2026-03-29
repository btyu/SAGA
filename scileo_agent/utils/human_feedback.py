"""
Utilities for collecting human feedback during optimization.

This module provides utilities for interactive human feedback collection,
including multi-line input, JSON validation, and confirmation prompts.
"""

import json
import sys
from typing import Dict, Any, Optional, Callable
from .logging import get_logger

logger = get_logger()

# Try to import IPython for better Jupyter support
try:
    from IPython import get_ipython
    from IPython.display import display, clear_output
    import ipywidgets as widgets
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False


def _is_running_in_jupyter():
    """Check if code is running in a Jupyter notebook."""
    if not IPYTHON_AVAILABLE:
        return False
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


async def get_multiline_input(prompt: str, end_marker: str = "END", allow_unchanged: bool = False) -> str:
    """
    Get multi-line input from the user.

    Args:
        prompt: The prompt to display to the user
        end_marker: The marker string that indicates the end of input (default: "END")
        allow_unchanged: If True, allow "<UNCHANGED>" as a single-line shortcut

    Returns:
        The complete multi-line input as a string
    """
    # Use IPython widgets if running in Jupyter
    if _is_running_in_jupyter():
        # Jupyter version is synchronous (ui_events requires it)
        # Call it directly - Python allows calling sync from async
        return _get_multiline_input_jupyter(prompt, end_marker, allow_unchanged)
    else:
        return _get_multiline_input_terminal(prompt, end_marker, allow_unchanged)


def _get_multiline_input_jupyter(prompt: str, end_marker: str = "END", allow_unchanged: bool = False) -> str:
    """
    Get multi-line input using IPython widgets (for Jupyter notebooks).

    Uses jupyter-ui-poll to allow UI events to be processed while waiting.

    Args:
        prompt: The prompt to display to the user
        end_marker: The marker string that indicates the end of input
        allow_unchanged: If True, allow "<UNCHANGED>" as a single-line shortcut

    Returns:
        The complete multi-line input as a string
    """
    import time
    from jupyter_ui_poll import ui_events

    print(f"\n{prompt}")
    if allow_unchanged:
        print(f"(Enter '<UNCHANGED>' to keep original, or enter your text and click Submit)\n")
    else:
        print(f"(Enter your text and click Submit when finished)\n")

    # Create text area widget
    textarea = widgets.Textarea(
        value='',
        placeholder='Enter your text here...',
        description='',
        layout=widgets.Layout(width='100%', height='300px')
    )

    # Create submit button
    submit_button = widgets.Button(
        description='Submit',
        button_style='success',
        tooltip='Click to submit your input',
        icon='check'
    )

    # Create output area for feedback
    output = widgets.Output()

    # Store the result
    result = {'value': None, 'submitted': False}

    def on_submit_clicked(b):
        with output:
            clear_output()
            result['value'] = textarea.value
            result['submitted'] = True
            # Disable button and textarea after submission
            submit_button.disabled = True
            textarea.disabled = True
            submit_button.description = 'Submitted ✓'
            print("✓ Input submitted successfully!")

    submit_button.on_click(on_submit_clicked)

    # Display the widgets
    display(textarea)
    display(submit_button)
    display(output)

    # Wait for submission using ui_events (synchronous blocking)
    print("\nWaiting for input submission...")
    print("⚠️  IMPORTANT: Do NOT run other cells while waiting for input!")
    print("    Running other cells will cause errors.")

    # Use synchronous ui_events to allow widget callbacks
    # Note: This may allow other cells to run, which will cause session errors
    with ui_events() as poll:
        while not result['submitted']:
            poll(10)  # Process up to 10 UI events
            time.sleep(0.05)  # Short sleep to stay responsive

    return result['value']


def _get_multiline_input_terminal(prompt: str, end_marker: str = "END", allow_unchanged: bool = False) -> str:
    """
    Get multi-line input using standard input (for terminal/console).

    Args:
        prompt: The prompt to display to the user
        end_marker: The marker string that indicates the end of input
        allow_unchanged: If True, allow "<UNCHANGED>" as a single-line shortcut

    Returns:
        The complete multi-line input as a string
    """
    print(f"\n{prompt}")
    if allow_unchanged:
        print(f"(Type '<UNCHANGED>' alone to keep original, or '{end_marker}' on a new line when finished)\n")
    else:
        print(f"(Type '{end_marker}' on a new line when finished)\n")

    lines = []
    while True:
        try:
            line = input()
            # Check for <UNCHANGED> on first line if allowed
            if allow_unchanged and len(lines) == 0 and line.strip() == "<UNCHANGED>":
                return "<UNCHANGED>"
            if line.strip() == end_marker:
                break
            lines.append(line)
        except EOFError:
            break

    return "\n".join(lines)


def validate_json(json_str: str) -> tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Validate a JSON string.

    Args:
        json_str: The JSON string to validate

    Returns:
        Tuple of (is_valid, parsed_dict, error_message)
    """
    try:
        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            return False, None, "JSON must be a dictionary/object, not a list or primitive type"
        return True, parsed, None
    except json.JSONDecodeError as e:
        return False, None, f"JSON parsing error: {str(e)}"


async def confirm_input(message: str = "Is this correct?") -> bool:
    """
    Ask the user to confirm their input.

    Args:
        message: The confirmation message to display

    Returns:
        True if user confirms, False otherwise
    """
    # Use IPython widgets if running in Jupyter
    if _is_running_in_jupyter():
        # Jupyter version is synchronous (ui_events requires it)
        # Call it directly - Python allows calling sync from async
        return _confirm_input_jupyter(message)
    else:
        return _confirm_input_terminal(message)


def _confirm_input_jupyter(message: str = "Is this correct?") -> bool:
    """
    Ask for confirmation using IPython widgets (for Jupyter notebooks).

    Args:
        message: The confirmation message to display

    Returns:
        True if user confirms, False otherwise
    """
    import time
    from jupyter_ui_poll import ui_events

    print(f"\n{message}")

    # Create Yes/No buttons
    yes_button = widgets.Button(
        description='Yes',
        button_style='success',
        tooltip='Confirm',
        icon='check'
    )

    no_button = widgets.Button(
        description='No',
        button_style='danger',
        tooltip='Cancel',
        icon='times'
    )

    # Create output area
    output = widgets.Output()

    # Store the result
    result = {'value': None, 'chosen': False}

    def on_yes_clicked(b):
        with output:
            clear_output()
            result['value'] = True
            result['chosen'] = True
            yes_button.disabled = True
            no_button.disabled = True
            print("✓ Confirmed")

    def on_no_clicked(b):
        with output:
            clear_output()
            result['value'] = False
            result['chosen'] = True
            yes_button.disabled = True
            no_button.disabled = True
            print("✗ Cancelled")

    yes_button.on_click(on_yes_clicked)
    no_button.on_click(on_no_clicked)

    # Display buttons side by side
    button_box = widgets.HBox([yes_button, no_button])
    display(button_box)
    display(output)

    print("⚠️  IMPORTANT: Do NOT run other cells while waiting for input!")

    # Wait for choice using synchronous ui_events
    with ui_events() as poll:
        while not result['chosen']:
            poll(10)  # Process up to 10 UI events
            time.sleep(0.05)  # Short sleep to stay responsive

    return result['value']


def _confirm_input_terminal(message: str = "Is this correct?") -> bool:
    """
    Ask for confirmation using standard input (for terminal/console).

    Args:
        message: The confirmation message to display

    Returns:
        True if user confirms, False otherwise
    """
    while True:
        response = input(f"\n{message} (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Please enter 'yes' or 'no'")


async def get_validated_json_input(
    prompt: str,
    validator: Optional[Callable[[Dict[str, Any]], tuple[bool, Optional[str]]]] = None,
    end_marker: str = "END",
    max_attempts: int = 5
) -> Dict[str, Any]:
    """
    Get validated JSON input from the user with retry logic.

    Args:
        prompt: The prompt to display to the user
        validator: Optional custom validator function that takes parsed dict and returns (is_valid, error_message)
        end_marker: The marker string that indicates the end of input
        max_attempts: Maximum number of attempts allowed

    Returns:
        The validated parsed JSON dictionary

    Raises:
        ValueError: If max attempts reached without valid input
    """
    attempt = 0

    while attempt < max_attempts:
        attempt += 1

        # Get multi-line input
        user_input = await get_multiline_input(prompt, end_marker)

        # Show what was entered
        print("\n" + "="*80)
        print("You entered:")
        print("-"*80)
        print(user_input)
        print("="*80)

        # Validate JSON format
        is_valid_json, parsed_dict, json_error = validate_json(user_input)

        if not is_valid_json:
            print(f"\n❌ Invalid JSON: {json_error}")
            if attempt < max_attempts:
                retry = input(f"\nWould you like to try again? (yes/no, attempts remaining: {max_attempts - attempt}): ").strip().lower()
                if retry not in ['yes', 'y']:
                    raise ValueError(f"User cancelled input after JSON validation error: {json_error}")
                continue
            else:
                raise ValueError(f"Maximum attempts ({max_attempts}) reached. Last error: {json_error}")

        # Apply custom validator if provided
        if validator is not None:
            is_valid_custom, custom_error = validator(parsed_dict)
            if not is_valid_custom:
                print(f"\n❌ Validation error: {custom_error}")
                if attempt < max_attempts:
                    retry = input(f"\nWould you like to try again? (yes/no, attempts remaining: {max_attempts - attempt}): ").strip().lower()
                    if retry not in ['yes', 'y']:
                        raise ValueError(f"User cancelled input after validation error: {custom_error}")
                    continue
                else:
                    raise ValueError(f"Maximum attempts ({max_attempts}) reached. Last error: {custom_error}")

        # Show formatted JSON for confirmation
        print("\n" + "="*80)
        print("Formatted JSON:")
        print("-"*80)
        print(json.dumps(parsed_dict, indent=2))
        print("="*80)

        # Ask for confirmation
        if await confirm_input("Do you confirm this input?"):
            logger.info("Human feedback received and confirmed")
            return parsed_dict
        else:
            print("\nInput not confirmed.")
            if attempt < max_attempts:
                retry = input(f"\nWould you like to try again? (yes/no, attempts remaining: {max_attempts - attempt}): ").strip().lower()
                if retry not in ['yes', 'y']:
                    raise ValueError("User cancelled input during confirmation")
                continue
            else:
                raise ValueError(f"Maximum attempts ({max_attempts}) reached without confirmation")

    raise ValueError(f"Maximum attempts ({max_attempts}) reached without valid input")


def display_objectives_for_feedback(objectives_dict: Dict[str, Any]) -> None:
    """
    Display objectives in a human-friendly format for feedback.

    Args:
        objectives_dict: The objectives dictionary from the LLM response
    """
    print("\n" + "="*80)
    print("LLM PROPOSED OBJECTIVES")
    print("="*80)

    if "reasoning" in objectives_dict:
        print(f"\nOverall Reasoning:")
        print(f"  {objectives_dict['reasoning']}")

    print(f"\nProposed Objectives ({len(objectives_dict.get('objectives', []))}):")
    print("-"*80)

    for i, obj in enumerate(objectives_dict.get('objectives', []), 1):
        print(f"\n{i}. {obj.get('name', 'N/A')}")
        if 'type' in obj:
            print(f"   Type: {obj['type']}")
        if 'description' in obj:
            print(f"   Description: {obj['description']}")
        if 'optimization_direction' in obj:
            print(f"   Direction: {obj['optimization_direction']}")
        if 'weight' in obj:
            print(f"   Weight: {obj['weight']}")
        if 'reasoning' in obj:
            print(f"   Reasoning: {obj['reasoning']}")

    print("\n" + "="*80)
    print("JSON FORMAT:")
    print("-"*80)
    print(json.dumps(objectives_dict, indent=2))
    print("="*80)


def validate_objectives_dict(objectives_dict: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate the structure of an objectives dictionary.

    Args:
        objectives_dict: The objectives dictionary to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for required top-level keys
    if "objectives" not in objectives_dict:
        return False, "Missing required key: 'objectives'"

    if not isinstance(objectives_dict["objectives"], list):
        return False, "'objectives' must be a list"

    if len(objectives_dict["objectives"]) == 0:
        return False, "'objectives' list cannot be empty"

    if "reasoning" not in objectives_dict:
        return False, "Missing required key: 'reasoning'"

    if not isinstance(objectives_dict["reasoning"], str):
        return False, "'reasoning' must be a string"

    # Check each objective
    for i, obj in enumerate(objectives_dict["objectives"]):
        if not isinstance(obj, dict):
            return False, f"Objective {i+1} must be a dictionary"

        # Check required fields
        required_fields = ["name", "description"]
        for field in required_fields:
            if field not in obj:
                return False, f"Objective {i+1} missing required field: '{field}'"

        # Check that name is a string
        if not isinstance(obj["name"], str) or not obj["name"].strip():
            return False, f"Objective {i+1} 'name' must be a non-empty string"

        # Check that description is a string
        if not isinstance(obj["description"], str) or not obj["description"].strip():
            return False, f"Objective {i+1} 'description' must be a non-empty string"

        # If type is present, validate it
        if "type" in obj:
            valid_types = ["candidate-wise", "population-wise", "filter"]
            if obj["type"] not in valid_types:
                return False, f"Objective {i+1} has invalid type: '{obj['type']}'. Must be one of {valid_types}"

        # If optimization_direction is present, validate it
        if "optimization_direction" in obj:
            valid_directions = ["maximize", "minimize"]
            if obj["optimization_direction"] not in valid_directions:
                return False, f"Objective {i+1} has invalid optimization_direction: '{obj['optimization_direction']}'. Must be one of {valid_directions}"

        # Filter objectives should not have optimization_direction
        if obj.get("type") == "filter" and "optimization_direction" in obj:
            return False, f"Objective {i+1} is a filter but has optimization_direction (filters should not have this field)"

        # Non-filter objectives should have optimization_direction
        if obj.get("type") in ["candidate-wise", "population-wise"] and "optimization_direction" not in obj:
            return False, f"Objective {i+1} (type: {obj['type']}) must have optimization_direction"

        # If weight is present, validate it
        if "weight" in obj:
            if not isinstance(obj["weight"], (int, float)):
                return False, f"Objective {i+1} 'weight' must be a number"

    return True, None


async def get_human_feedback_on_objectives(
    llm_proposed_objectives: Dict[str, Any],
    end_marker: str = "<END>",
    max_attempts: int = 5,
    support_population_wise: bool = False,
    support_filter: bool = False,
    requires_weights: bool = False
) -> Dict[str, Any]:
    """
    Get human feedback on LLM-proposed objectives.

    This function displays the LLM's proposed objectives and allows the human
    to either accept the proposal (via button) or provide a revised version in JSON format.

    Args:
        llm_proposed_objectives: The objectives dictionary proposed by the LLM
        end_marker: The marker string that indicates the end of input (default: "<END>")
        max_attempts: Maximum number of attempts allowed
        support_population_wise: Whether population-wise objectives are supported
        support_filter: Whether filter objectives are supported
        requires_weights: Whether objective weights are required

    Returns:
        The human-revised objectives dictionary (or original if accepted)
    """
    # Use different UI based on environment
    if _is_running_in_jupyter():
        return await _get_human_feedback_jupyter(
            llm_proposed_objectives,
            end_marker,
            max_attempts,
            support_population_wise,
            support_filter,
            requires_weights
        )
    else:
        # Display the LLM's proposed objectives for terminal (only once)
        display_objectives_for_feedback(llm_proposed_objectives)
        return await _get_human_feedback_terminal(
            llm_proposed_objectives,
            end_marker,
            max_attempts,
            support_population_wise,
            support_filter,
            requires_weights
        )


async def _get_human_feedback_jupyter(
    llm_proposed_objectives: Dict[str, Any],
    end_marker: str,
    max_attempts: int,
    support_population_wise: bool,
    support_filter: bool,
    requires_weights: bool
) -> Dict[str, Any]:
    """
    Get human feedback using Jupyter widgets (button-based UI).

    First asks if user wants to accept the LLM's proposal.
    If not, prompts for revision with confirmation.
    """
    import time
    from jupyter_ui_poll import ui_events

    # Build dynamic prompt based on configuration
    obj_fields = ['      "name": "objective_name",']

    if support_population_wise or support_filter:
        type_options = ["candidate-wise"]
        if support_population_wise:
            type_options.append("population-wise")
        if support_filter:
            type_options.append("filter")
        type_str = " or ".join([f'"{t}"' for t in type_options])
        obj_fields.append(f'      "type": {type_str},')

    obj_fields.append('      "description": "detailed description",')

    if support_filter:
        obj_fields.append('      "optimization_direction": "maximize" or "minimize" (omit for filter objectives),')
    else:
        obj_fields.append('      "optimization_direction": "maximize" or "minimize",')

    if requires_weights:
        obj_fields.append('      "weight": 1.0,')

    obj_fields.append('      "reasoning": "why this objective is important"')
    obj_structure = '\n'.join(obj_fields)

    # Keep trying until we get valid input
    first_time = True
    while True:
        if not first_time:
            # Only show header on subsequent loops (after cancellation)
            print("\n" + "="*80)
            print("LLM PROPOSED OBJECTIVES - REVIEW")
            print("="*80)
        first_time = False

        # Step 1: Ask if user wants to accept the LLM's proposal
        accept_proposal = _confirm_with_preview_jupyter(
            "Accept the LLM's proposed objectives?",
            llm_proposed_objectives
        )

        if accept_proposal:
            print("\n✓ Using LLM's proposed objectives")
            logger.info("Human accepted LLM proposal without changes")
            return llm_proposed_objectives

        # Step 2: User wants to revise - get revised objectives
        print("\n" + "="*80)
        print("PROVIDE REVISED OBJECTIVES")
        print("="*80)
        print(f"\nEnter your revised objectives in JSON format:")
        print(f"\nExpected structure:")
        print(f"{{")
        print(f'  "objectives": [')
        print(f'    {{')
        print(obj_structure)
        print(f'    }}')
        print(f'  ],')
        print(f'  "reasoning": "overall justification for the objectives"')
        print(f"}}")
        print(f"\nEnter your JSON below and click Submit:")

        # Get revised objectives with validation and confirmation
        revised_dict = _get_revised_objectives_jupyter(obj_structure)

        if revised_dict is not None:
            # Successfully got confirmed revision
            logger.info("Human submitted revised objectives")
            return revised_dict
        else:
            # User cancelled - loop back to ask about accepting LLM proposal again
            print("\n✗ Revision cancelled. Returning to review...")
            continue


def _confirm_with_preview_jupyter(message: str, objectives_dict: Dict[str, Any]) -> bool:
    """
    Show objectives preview and ask for confirmation with Yes/No buttons.

    Args:
        message: The confirmation message
        objectives_dict: The objectives to preview

    Returns:
        True if user confirms, False otherwise
    """
    import time
    from jupyter_ui_poll import ui_events

    print(f"\n{message}")
    print("\n" + "-"*80)
    display_objectives_for_feedback(objectives_dict)
    print("-"*80)

    # Create Yes/No buttons
    yes_button = widgets.Button(
        description='Yes - Accept',
        button_style='success',
        tooltip='Accept these objectives',
        icon='check',
        layout=widgets.Layout(width='200px', height='50px')
    )

    no_button = widgets.Button(
        description='No - Revise',
        button_style='primary',
        tooltip='Provide revised objectives',
        icon='edit',
        layout=widgets.Layout(width='200px', height='50px')
    )

    # Create output area
    output = widgets.Output()

    # Store the result
    result = {'value': None, 'chosen': False}

    def on_yes_clicked(b):
        with output:
            clear_output()
            result['value'] = True
            result['chosen'] = True
            yes_button.disabled = True
            no_button.disabled = True
            print("✓ Accepted")

    def on_no_clicked(b):
        with output:
            clear_output()
            result['value'] = False
            result['chosen'] = True
            yes_button.disabled = True
            no_button.disabled = True
            print("→ Proceeding to revision...")

    yes_button.on_click(on_yes_clicked)
    no_button.on_click(on_no_clicked)

    # Display buttons side by side
    button_box = widgets.HBox([yes_button, no_button])
    display(button_box)
    display(output)

    print("\n⚠️  IMPORTANT: Do NOT run other cells while waiting for input!")

    # Wait for choice using synchronous ui_events
    with ui_events() as poll:
        while not result['chosen']:
            poll(10)  # Process up to 10 UI events
            time.sleep(0.05)  # Short sleep to stay responsive

    return result['value']


def _get_revised_objectives_jupyter(obj_structure: str) -> Optional[Dict[str, Any]]:
    """
    Get revised objectives with text area, validation, and confirmation.

    Args:
        obj_structure: The objective structure template

    Returns:
        Validated objectives dict, or None if user cancelled
    """
    import time
    from jupyter_ui_poll import ui_events

    # Create text area for revisions
    textarea = widgets.Textarea(
        value='',
        placeholder=f'''Enter revised objectives in JSON format:
{{
  "objectives": [
    {{
{obj_structure}
    }}
  ],
  "reasoning": "overall justification for the objectives"
}}''',
        description='',
        layout=widgets.Layout(width='100%', height='300px')
    )

    # Create submit button
    submit_button = widgets.Button(
        description='Submit',
        button_style='primary',
        tooltip='Submit your revised objectives',
        icon='check',
        layout=widgets.Layout(width='200px', height='50px')
    )

    # Create cancel button
    cancel_button = widgets.Button(
        description='Cancel',
        button_style='warning',
        tooltip='Cancel and go back',
        icon='times',
        layout=widgets.Layout(width='200px', height='50px')
    )

    # Create output area
    output = widgets.Output()

    # Store the result
    result = {'value': None, 'submitted': False, 'cancelled': False}

    def on_submit_clicked(b):
        with output:
            clear_output()
            user_input = textarea.value.strip()

            if not user_input:
                print("❌ Please enter revised objectives in the text area")
                return

            # Validate JSON format
            is_valid_json, parsed_dict, json_error = validate_json(user_input)

            if not is_valid_json:
                print(f"❌ Invalid JSON: {json_error}")
                print("Please fix the JSON and try again.")
                return

            # Apply custom validator
            is_valid_custom, custom_error = validate_objectives_dict(parsed_dict)
            if not is_valid_custom:
                print(f"❌ Validation error: {custom_error}")
                print("Please fix the objectives and try again.")
                return

            # Show formatted JSON and ask for confirmation
            print("Your Revised Objectives:")
            print("-"*80)
            print(json.dumps(parsed_dict, indent=2))
            print("-"*80)

            # Ask for final confirmation using nested widget approach
            confirmed = _final_confirmation_jupyter("Confirm these revised objectives?")

            if confirmed:
                result['value'] = parsed_dict
                result['submitted'] = True
                # Disable all controls
                submit_button.disabled = True
                cancel_button.disabled = True
                textarea.disabled = True
                submit_button.button_style = 'success'
                submit_button.description = 'Submitted ✓'
                print("\n✓ Revision confirmed and submitted!")
            else:
                print("\n✗ Not confirmed. You can edit and try again.")

    def on_cancel_clicked(b):
        with output:
            clear_output()
            result['cancelled'] = True
            # Disable all controls
            submit_button.disabled = True
            cancel_button.disabled = True
            textarea.disabled = True
            print("✗ Cancelled. Returning to review...")

    submit_button.on_click(on_submit_clicked)
    cancel_button.on_click(on_cancel_clicked)

    # Display the UI
    display(textarea)
    button_box = widgets.HBox([submit_button, cancel_button])
    display(button_box)
    display(output)

    print("\n⚠️  IMPORTANT: Do NOT run other cells while waiting for input!")

    # Wait for submission or cancellation
    with ui_events() as poll:
        while not (result['submitted'] or result['cancelled']):
            poll(10)  # Process up to 10 UI events
            time.sleep(0.05)  # Short sleep to stay responsive

    return result['value']  # Returns None if cancelled


def _final_confirmation_jupyter(message: str) -> bool:
    """
    Final confirmation with Yes/No buttons (nested within another widget callback).

    Args:
        message: The confirmation message

    Returns:
        True if confirmed, False otherwise
    """
    import time
    from jupyter_ui_poll import ui_events

    print(f"\n{message}")

    # Create Yes/No buttons
    yes_button = widgets.Button(
        description='Yes',
        button_style='success',
        icon='check'
    )

    no_button = widgets.Button(
        description='No',
        button_style='danger',
        icon='times'
    )

    output = widgets.Output()
    result = {'value': None, 'chosen': False}

    def on_yes_clicked(b):
        with output:
            clear_output()
            result['value'] = True
            result['chosen'] = True
            yes_button.disabled = True
            no_button.disabled = True
            print("✓ Confirmed")

    def on_no_clicked(b):
        with output:
            clear_output()
            result['value'] = False
            result['chosen'] = True
            yes_button.disabled = True
            no_button.disabled = True
            print("✗ Not confirmed")

    yes_button.on_click(on_yes_clicked)
    no_button.on_click(on_no_clicked)

    button_box = widgets.HBox([yes_button, no_button])
    display(button_box)
    display(output)

    # Wait for choice
    with ui_events() as poll:
        while not result['chosen']:
            poll(10)
            time.sleep(0.05)

    return result['value']


async def _get_human_feedback_terminal(
    llm_proposed_objectives: Dict[str, Any],
    end_marker: str,
    max_attempts: int,
    support_population_wise: bool,
    support_filter: bool,
    requires_weights: bool
) -> Dict[str, Any]:
    """
    Get human feedback using terminal interface.

    Provides options via text prompts.
    """
    # Build dynamic prompt based on configuration
    obj_fields = ['      "name": "objective_name",']

    if support_population_wise or support_filter:
        type_options = ["candidate-wise"]
        if support_population_wise:
            type_options.append("population-wise")
        if support_filter:
            type_options.append("filter")
        type_str = " or ".join([f'"{t}"' for t in type_options])
        obj_fields.append(f'      "type": {type_str},')

    obj_fields.append('      "description": "detailed description",')

    if support_filter:
        obj_fields.append('      "optimization_direction": "maximize" or "minimize" (omit for filter objectives),')
    else:
        obj_fields.append('      "optimization_direction": "maximize" or "minimize",')

    if requires_weights:
        obj_fields.append('      "weight": 1.0,')

    obj_fields.append('      "reasoning": "why this objective is important"')
    obj_structure = '\n'.join(obj_fields)

    # Ask if user wants to accept or revise
    print("\n" + "="*80)
    print("CHOOSE AN OPTION:")
    print("="*80)
    print("1. Accept the LLM's proposal as-is")
    print("2. Provide revised objectives")
    print("="*80)

    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        if choice == "1":
            print("\n" + "="*80)
            print("Using LLM's original proposal (unchanged)")
            print("="*80)
            logger.info("Human accepted LLM proposal without changes")
            return llm_proposed_objectives
        elif choice == "2":
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    # User chose to revise - get revised objectives
    prompt = f"""
Please provide your revised objectives in JSON format.
You can modify, add, or remove objectives as needed.

The JSON should follow this structure:
{{
  "objectives": [
    {{
{obj_structure}
    }}
  ],
  "reasoning": "overall justification for the objectives"
}}

Enter your revised objectives below (type '{end_marker}' on a new line when finished):"""

    # Get input with validation loop
    attempt = 0
    while attempt < max_attempts:
        attempt += 1

        # Get multi-line input
        user_input = await get_multiline_input(prompt, end_marker, allow_unchanged=False)

        # Show what was entered
        print("\n" + "="*80)
        print("You entered:")
        print("-"*80)
        print(user_input)
        print("="*80)

        # Validate JSON format
        is_valid_json, parsed_dict, json_error = validate_json(user_input)

        if not is_valid_json:
            print(f"\n❌ Invalid JSON: {json_error}")
            print("Please try again.")
            continue

        # Apply custom validator
        is_valid_custom, custom_error = validate_objectives_dict(parsed_dict)
        if not is_valid_custom:
            print(f"\n❌ Validation error: {custom_error}")
            print("Please try again.")
            continue

        # Show formatted JSON for confirmation
        print("\n" + "="*80)
        print("Formatted JSON:")
        print("-"*80)
        print(json.dumps(parsed_dict, indent=2))
        print("="*80)

        # Ask for confirmation
        if await confirm_input("Do you confirm this input?"):
            logger.info("Human feedback received and confirmed")
            return parsed_dict
        else:
            print("\nInput not confirmed.")
            if attempt < max_attempts:
                retry = input(f"\nWould you like to try again? (yes/no, attempts remaining: {max_attempts - attempt}): ").strip().lower()
                if retry not in ['yes', 'y']:
                    raise ValueError("User cancelled input during confirmation")
                continue
            else:
                raise ValueError(f"Maximum attempts ({max_attempts}) reached without confirmation")

    raise ValueError(f"Maximum attempts ({max_attempts}) reached without valid input")
