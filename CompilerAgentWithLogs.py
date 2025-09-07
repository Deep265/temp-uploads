import logging
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, List
import os
import re
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('project_fixer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@tool
def read_file(file_path: str) -> str:
    """Read the contents of a file in the project directory.
    
    Args:
        file_path (str): Relative path to the file.
    
    Returns:
        str: File contents or error message if file cannot be read.
    """
    logger.info(f"Tool 'read_file' called with file_path: {file_path}")
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        logger.info(f"Tool 'read_file' response: Successfully read {len(content)} characters")
        return content
    except Exception as e:
        logger.error(f"Tool 'read_file' error: {str(e)}")
        return f"Error reading file: {str(e)}"

@tool
def analyze_dependency(dependency: str) -> str:
    """Analyze a Maven or Gradle dependency for version or availability.
    
    Args:
        dependency (str): Dependency in format 'groupId:artifactId' (Maven) or 'group:artifact' (Gradle).
    
    Returns:
        str: Information about the dependency or error message.
    """
    logger.info(f"Tool 'analyze_dependency' called with dependency: {dependency}")
    try:
        result = f"Dependency {dependency} is valid. Latest version: placeholder-version"
        logger.info(f"Tool 'analyze_dependency' response: {result}")
        return result
    except Exception as e:
        logger.error(f"Tool 'analyze_dependency' error: {str(e)}")
        return f"Error analyzing dependency: {str(e)}"

class State(TypedDict):
    """State schema for the LangGraph workflow."""
    project_dir: str
    iteration: int
    max_iterations: int
    build_result: Dict[str, Any]
    project_files: Dict[str, str]
    ai_response: Dict[str, Any]
    status: str

class LangGraphProjectFixer:
    """A LangGraph-based project fixer for Spring Boot projects using a multi-agent workflow."""
    
    def __init__(self, project_dir: str):
        """
        Initialize the LangGraphProjectFixer with ChatOpenAI for LLM calls.

        Args:
            project_dir (str): Path to the Spring Boot project directory.

        Raises:
            ValueError: If neither pom.xml nor build.gradle is found in the project directory.
        """
        logger.info(f"Initializing LangGraphProjectFixer for project directory: {project_dir}")
        self.project_dir = project_dir
        self.build_tool = self._detect_build_tool()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.tools = [read_file, analyze_dependency]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        graph = StateGraph(State)
        graph.add_node("compile", self.compile_node)
        graph.add_node("fixer", self.fixer_node)
        graph.add_node("writer", self.writer_node)
        graph.set_entry_point("compile")
        graph.add_conditional_edges(
            "compile",
            self.decide_next,
            {"continue": "fixer", "end": END}
        )
        graph.add_edge("fixer", "writer")
        graph.add_edge("writer", "compile")
        self.app = graph.compile()
        logger.info("LangGraph workflow initialized")

    def _detect_build_tool(self) -> str:
        """
        Detect the build tool used in the project (Maven or Gradle).

        Returns:
            str: The detected build tool ('maven' or 'gradle').

        Raises:
            ValueError: If no supported build tool is detected.
        """
        logger.info("Detecting build tool")
        if os.path.exists(os.path.join(self.project_dir, 'pom.xml')):
            logger.info("Detected Maven project (pom.xml found)")
            return 'maven'
        elif os.path.exists(os.path.join(self.project_dir, 'build.gradle')):
            logger.info("Detected Gradle project (build.gradle found)")
            return 'gradle'
        logger.error("No supported build tool detected")
        raise ValueError("Unsupported build tool")

    def _run_build(self) -> Dict[str, Any]:
        """
        Run the build command for the project using the detected build tool.

        Returns:
            Dict[str, Any]: Build result with success status, output, and parsed errors.
        """
        cmd = 'mvn clean install' if self.build_tool == 'maven' else 'gradle build'
        logger.info(f"Running build command: {cmd}")
        output_file = os.path.join(self.project_dir, 'build_output.txt')
        full_cmd = f"{cmd} > {output_file} 2>&1"
        return_code = os.system(full_cmd)
        
        try:
            with open(output_file, 'r') as f:
                output = f.read()
            logger.info(f"Build output captured: {len(output)} characters")
        except FileNotFoundError:
            output = "Error: Build output file not generated."
            logger.error(output)
        
        if os.path.exists(output_file):
            os.remove(output_file)
            logger.info("Temporary build output file removed")
        
        errors = self._parse_errors(output)
        logger.info(f"Parsed {len(errors)} build errors")
        return {
            "success": return_code == 0,
            "output": output,
            "errors": errors
        }

    def _parse_errors(self, output: str) -> List[Dict[str, Any]]:
        """
        Parse build output to extract compilation errors or warnings.

        Args:
            output (str): Raw build output from Maven or Gradle.

        Returns:
            List[Dict[str, Any]]: List of parsed errors with file, line, type, and message.
        """
        logger.info("Parsing build output for errors")
        errors = []
        for line in output.splitlines():
            match = re.match(r'(\S+):(\d+): (error|warning): (.+)', line)
            if match:
                file, line, typ, msg = match.groups()
                error = {"file": file, "line": int(line), "type": typ.capitalize(), "message": msg}
                errors.append(error)
                logger.debug(f"Parsed error: {error}")
        return errors

    def _get_project_files(self) -> Dict[str, str]:
        """
        Collect relevant project files (Java, XML, YAML, properties, Gradle).

        Returns:
            Dict[str, str]: Dictionary of file paths (relative) to their contents.
        """
        logger.info("Collecting project files")
        files = {}
        for root, _, filenames in os.walk(self.project_dir):
            for fname in filenames:
                if fname.endswith(('.java', '.xml', '.yml', '.properties', '.gradle')):
                    path = os.path.join(root, fname)
                    with open(path, 'r') as f:
                        files[os.path.relpath(path, self.project_dir)] = f.read()
        logger.info(f"Collected {len(files)} project files")
        return files

    def _apply_fixes(self, corrected_files: Dict[str, str]):
        """
        Apply fixed file contents to the project directory.

        Args:
            corrected_files (Dict[str, str]): Dictionary of file paths to their corrected contents.
        """
        logger.info(f"Applying fixes to {len(corrected_files)} files")
        for rel_path, content in corrected_files.items():
            path = os.path.join(self.project_dir, rel_path)
            with open(path, 'w') as f:
                f.write(content)
            logger.info(f"Updated file: {rel_path}")

    def _format_llm_prompt(self, input_data: Dict[str, Any]) -> str:
        """
        Format the prompt for the LLM with project details and errors.

        Args:
            input_data (Dict[str, Any]): Project files, errors, and build output.

        Returns:
            str: Formatted prompt string for the LLM.
        """
        logger.info("Formatting LLM prompt")
        return f"""Analyze and fix the following Spring Boot project using available tools:
Files: {json.dumps(input_data['files'])}
Errors: {json.dumps(input_data['errors'])}
Build Output: {input_data['build_output']}

Use tools like 'read_file' to access file contents or 'analyze_dependency' to check dependencies.
Follow the multi-agent workflow: Compiler -> Fixer -> Writer.
Output in JSON format:
{{
  "status": "Fixed|Partial Fix|Unresolvable",
  "errors": [
    {{"file": "string", "line": int, "message": "string", "type": "Syntax|Config|Runtime|Dependency", "cause": "string"}}
  ],
  "fixes": [
    {{"file": "string", "original_line": int, "fix_description": "string", "updated_code_snippet": "string"}}
  ],
  "corrected_files": {{"file_name.java": "full updated code", ...}},
  "verification": "string",
  "suggestions": "string"
}}"""

    def compile_node(self, state: State) -> State:
        """
        Compiler agent node: Run build and collect errors.

        Args:
            state (State): Current workflow state.

        Returns:
            State: Updated state with build result and project files (if needed).
        """
        logger.info(f"Starting compile node (iteration {state['iteration']})")
        state["build_result"] = self._run_build()
        if state["build_result"]["success"]:
            state["status"] = "No errors found"
            logger.info("Build successful, no errors found")
        else:
            state["project_files"] = self._get_project_files()
            logger.info("Build failed, collected project files for analysis")
        return state

    def fixer_node(self, state: State) -> State:
        """
        Fixer agent node: Analyze errors and request fixes from the LLM with tool calling.

        Args:
            state (State): Current workflow state with build result.

        Returns:
            State: Updated state with LLM response or unresolvable status.
        """
        logger.info(f"Starting fixer node (iteration {state['iteration']})")
        input_data = {
            "files": state["project_files"],
            "errors": state["build_result"]["errors"],
            "build_output": state["build_result"]["output"]
        }
        llm_prompt = self._format_llm_prompt(input_data)
        messages = [
            SystemMessage(content="You are an expert in fixing Spring Boot projects. Use provided tools to analyze and fix errors. Always respond in the specified JSON format."),
            HumanMessage(content=llm_prompt)
        ]
        response = self.llm_with_tools.invoke(messages)
        
        # Handle tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            logger.info(f"LLM requested {len(response.tool_calls)} tool calls")
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                tool_result = self.tools[[t.name for t in self.tools].index(tool_name)].invoke(tool_args)
                logger.info(f"Tool {tool_name} returned: {tool_result}")
                messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call['id']))
            logger.info("Re-invoking LLM with tool results")
            final_response = self.llm_with_tools.invoke(messages)
            response_content = final_response.content
        else:
            response_content = response.content
            logger.info("No tool calls requested by LLM")

        try:
            fix_data = json.loads(response_content)
            state["ai_response"] = fix_data
            logger.info(f"LLM response parsed: status={fix_data.get('status')}")
            if fix_data.get("status") == "Unresolvable":
                state["status"] = "Unresolvable"
                logger.warning("LLM marked issue as unresolvable")
        except json.JSONDecodeError:
            state["status"] = "Unresolvable"
            logger.error("Failed to parse LLM response as JSON")
        return state

    def writer_node(self, state: State) -> State:
        """
        Writer agent node: Apply fixes to project files and increment iteration.

        Args:
            state (State): Current workflow state with LLM response.

        Returns:
            State: Updated state with incremented iteration count.
        """
        logger.info(f"Starting writer node (iteration {state['iteration']})")
        if "ai_response" in state and "corrected_files" in state["ai_response"]:
            self._apply_fixes(state["ai_response"]["corrected_files"])
        else:
            logger.warning("No corrected files to apply")
        state["iteration"] += 1
        logger.info(f"Iteration incremented to {state['iteration']}")
        return state

    def decide_next(self, state: State) -> str:
        """
        Decide the next step in the workflow based on build result and iteration count.

        Args:
            state (State): Current workflow state.

        Returns:
            str: Next node ('continue' or 'end').
        """
        logger.info(f"Deciding next step (iteration {state['iteration']})")
        if state["build_result"]["success"] or state["iteration"] >= state["max_iterations"] or state.get("status") == "Unresolvable":
            if state["build_result"]["success"] and state["status"] != "No errors found":
                state["status"] = "Fixed"
                logger.info("Project fixed successfully")
            elif state["iteration"] >= state["max_iterations"]:
                state["status"] = "Partial Fix"
                logger.warning("Max iterations reached")
            elif state["status"] == "Unresolvable":
                logger.warning("Workflow ending due to unresolvable issue")
            return "end"
        logger.info("Continuing to fixer node")
        return "continue"

    def fix_project(self) -> Dict[str, Any]:
        """
        Execute the LangGraph workflow to fix the Spring Boot project using the LLM.

        Returns:
            Dict[str, Any]: JSON-compatible result with status, message, and optional remaining errors.
        """
        logger.info("Starting fix_project workflow")
        initial_state: State = {
            "project_dir": self.project_dir,
            "iteration": 0,
            "max_iterations": 3,
            "build_result": None,
            "project_files": None,
            "ai_response": None,
            "status": None
        }
        final_state = self.app.invoke(initial_state)
        if final_state["status"] == "No errors found":
            logger.info("Workflow completed: No errors found")
            return {"status": "No errors found", "message": "Project compiles successfully."}
        elif final_state["status"] == "Unresolvable":
            logger.error("Workflow completed: Unresolvable issue")
            return {"status": "Unresolvable", "message": "Invalid LLM response or unresolvable issue"}
        elif final_state["status"] == "Partial Fix":
            remaining_errors = self._run_build()["errors"]
            logger.warning(f"Workflow completed: Partial fix with {len(remaining_errors)} remaining errors")
            return {"status": "Partial Fix", "message": "Max iterations reached", "remaining_errors": remaining_errors}
        else:
            logger.info("Workflow completed: Project fixed")
            return {"status": "Fixed"}

# Example usage
if __name__ == "__main__":
    fixer = LangGraphProjectFixer("/path/to/project")
    result = fixer.fix_project()
    print(json.dumps(result))
