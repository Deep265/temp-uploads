from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, List
import subprocess
import os
import re
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

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
    
    def __init__(self, project_dir: str, llm_call: callable):
        """
        Initialize the LangGraphProjectFixer with a callable for LLM calls.

        Args:
            project_dir (str): Path to the Spring Boot project directory.
            llm_call (callable): Function for LLM-based error analysis and fixes.

        Raises:
            ValueError: If neither pom.xml nor build.gradle is found in the project directory.
        """
        self.project_dir = project_dir
        self.llm_call = llm_call
        self.build_tool = self._detect_build_tool()
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

    def _detect_build_tool(self) -> str:
        """
        Detect the build tool used in the project (Maven or Gradle).

        Returns:
            str: The detected build tool ('maven' or 'gradle').

        Raises:
            ValueError: If no supported build tool is detected.
        """
        if os.path.exists(os.path.join(self.project_dir, 'pom.xml')):
            return 'maven'
        elif os.path.exists(os.path.join(self.project_dir, 'build.gradle')):
            return 'gradle'
        raise ValueError("Unsupported build tool")

    def _run_build(self) -> Dict[str, Any]:
        """
        Run the build command for the project using the detected build tool.

        Returns:
            Dict[str, Any]: Build result with success status, output, and parsed errors.
        """
        cmd = ['mvn', 'clean', 'install'] if self.build_tool == 'maven' else ['gradle', 'build']
        process = subprocess.run(cmd, cwd=self.project_dir, capture_output=True, text=True)
        output = process.stdout + process.stderr
        errors = self._parse_errors(output)
        return {
            "success": process.returncode == 0,
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
        errors = []
        for line in output.splitlines():
            match = re.match(r'(\S+):(\d+): (error|warning): (.+)', line)
            if match:
                file, line, typ, msg = match.groups()
                errors.append({"file": file, "line": int(line), "type": typ.capitalize(), "message": msg})
        return errors

    def _get_project_files(self) -> Dict[str, str]:
        """
        Collect relevant project files (Java, XML, YAML, properties, Gradle).

        Returns:
            Dict[str, str]: Dictionary of file paths (relative) to their contents.
        """
        files = {}
        for root, _, filenames in os.walk(self.project_dir):
            for fname in filenames:
                if fname.endswith(('.java', '.xml', '.yml', '.properties', '.gradle')):
                    path = os.path.join(root, fname)
                    with open(path, 'r') as f:
                        files[os.path.relpath(path, self.project_dir)] = f.read()
        return files

    def _apply_fixes(self, corrected_files: Dict[str, str]):
        """
        Apply fixed file contents to the project directory.

        Args:
            corrected_files (Dict[str, str]): Dictionary of file paths to their corrected contents.
        """
        for rel_path, content in corrected_files.items():
            path = os.path.join(self.project_dir, rel_path)
            with open(path, 'w') as f:
                f.write(content)

    def _format_llm_prompt(self, input_data: Dict[str, Any]) -> str:
        """
        Format the prompt for the LLM with project details and errors.

        Args:
            input_data (Dict[str, Any]): Project files, errors, and build output.

        Returns:
            str: Formatted prompt string for the LLM.
        """
        return f"""Analyze and fix the following Spring Boot project:
Files: {json.dumps(input_data['files'])}
Errors: {json.dumps(input_data['errors'])}
Build Output: {input_data['build_output']}

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
        state["build_result"] = self._run_build()
        if state["build_result"]["success"]:
            state["status"] = "No errors found"
        else:
            state["project_files"] = self._get_project_files()
        return state

    def fixer_node(self, state: State) -> State:
        """
        Fixer agent node: Analyze errors and request fixes from the LLM.

        Args:
            state (State): Current workflow state with build result.

        Returns:
            State: Updated state with LLM response or unresolvable status.
        """
        input_data = {
            "files": state["project_files"],
            "errors": state["build_result"]["errors"],
            "build_output": state["build_result"]["output"]
        }
        llm_prompt = self._format_llm_prompt(input_data)
        llm_response = self.llm_call(llm_prompt)
        try:
            fix_data = json.loads(llm_response)
            state["ai_response"] = fix_data
            if fix_data.get("status") == "Unresolvable":
                state["status"] = "Unresolvable"
        except json.JSONDecodeError:
            state["status"] = "Unresolvable"
        return state

    def writer_node(self, state: State) -> State:
        """
        Writer agent node: Apply fixes to project files and increment iteration.

        Args:
            state (State): Current workflow state with LLM response.

        Returns:
            State: Updated state with incremented iteration count.
        """
        if "ai_response" in state and "corrected_files" in state["ai_response"]:
            self._apply_fixes(state["ai_response"]["corrected_files"])
        state["iteration"] += 1
        return state

    def decide_next(self, state: State) -> str:
        """
        Decide the next step in the workflow based on build result and iteration count.

        Args:
            state (State): Current workflow state.

        Returns:
            str: Next node ('continue' or 'end').
        """
        if state["build_result"]["success"] or state["iteration"] >= state["max_iterations"] or state.get("status") == "Unresolvable":
            if state["build_result"]["success"] and state["status"] != "No errors found":
                state["status"] = "Fixed"
            elif state["iteration"] >= state["max_iterations"]:
                state["status"] = "Partial Fix"
            return "end"
        return "continue"

    def fix_project(self) -> Dict[str, Any]:
        """
        Execute the LangGraph workflow to fix the Spring Boot project using the LLM.

        Returns:
            Dict[str, Any]: JSON-compatible result with status, message, and optional remaining errors.
        """
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
            return {"status": "No errors found", "message": "Project compiles successfully."}
        elif final_state["status"] == "Unresolvable":
            return {"status": "Unresolvable", "message": "Invalid LLM response or unresolvable issue"}
        elif final_state["status"] == "Partial Fix":
            remaining_errors = self._run_build()["errors"]
            return {"status": "Partial Fix", "message": "Max iterations reached", "remaining_errors": remaining_errors}
        else:
            return {"status": "Fixed"}

# Example usage
if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    def llm_call(prompt: str) -> str:
        """
        LLM call function using ChatOpenAI.

        Args:
            prompt (str): Input prompt for LLM analysis.

        Returns:
            str: LLM response content.
        """
        messages = [
            SystemMessage(content="You are an expert in fixing Spring Boot projects. Always respond in the specified JSON format."),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        return response.content

    fixer = LangGraphProjectFixer("/path/to/project", llm_call)
    result = fixer.fix_project()
    print(json.dumps(result))
