# Avra AGI Project - Phase 1: Foundational Component - Gemini Integration (at0 concept)
# File: avra_at0_colab/memory_integrated_chat_v16.py
# Description: Major leap - Implements Tool Execution Feedback Loop using Chat Sessions for dynamic processing.

# To run this code in Google Colab, you need to:
# 1. Run the setup cells you provided:
#    !pip install -U -q "google"
#    !pip install -U -q "google.genai"
#    import os
#    from google.colab import userdata
#    os.environ["GEMINI_API_KEY"] = userdata.get("GOOGLE_API_KEY")
#    # Add drive mount/chdir if needed, especially if saving to Drive
# 2. Obtain a Gemini API key from Google AI Studio (https://aistudio.google.com/).
# 3. Add your API key as a Colab Secret named 'GOOGLE_API_KEY' (matching the userdata.get name).
# 4. Make sure your Colab environment has write access to the location where avra_memory.json will be saved (e.g., mounted Google Drive).

import os
import uuid # Used for generating unique identifiers
import textwrap # To format output nicely
import json # Needed for parsing model output, saving/loading
import re # Import regex for robust JSON extraction
import time # For potential delays in retry
import io # To capture stdout/stderr
import sys # To redirect stdout/stderr

# Import genai and types correctly
from google import genai
from google.genai import types

# --- Configuration ---
MEMORY_FILE = "avra_memory.json"
MODEL_NAME_TO_USE = "gemini-2.5-flash-preview-04-17"
MAX_EXTRACTION_RETRIES = 2
MAX_TOOL_EXECUTION_RETRIES = 2 # Conceptual tries if tool execution fails

# --- Avra Core Component: Knowledge Node ---
class KnowledgeNode:
    """Represents a single node or entity within Avra's Knowledge Graph."""
    def __init__(self, node_type: str, name: str, attributes: dict = None):
        if not isinstance(node_type, str) or not node_type.strip():
            raise ValueError("node_type must be a non-empty string.")
        if not isinstance(name, str) or not name.strip():
             raise ValueError("name must be a non-empty string.")
        if attributes is not None and not isinstance(attributes, dict):
             raise ValueError("attributes must be a dictionary or None.")

        self.id: str = str(uuid.uuid4())
        self.node_type: str = node_type.strip()
        self.name: str = name.strip()
        self.attributes: dict = attributes if attributes is not None else {}

    def add_attribute(self, key: str, value: any):
        if not isinstance(key, str) or not key.strip():
             print(f"Warning: Node {self.id}: Attempted to add attribute with invalid key.")
             return
        stripped_key = key.strip()
        self.attributes[stripped_key] = value

    def get_attribute(self, key: str) -> any:
        if not isinstance(key, str) or not key.strip():
             return None
        return self.attributes.get(key.strip(), None)

    def remove_attribute(self, key: str):
        if not isinstance(key, str) or not key.strip():
             print("Warning: Attempted to remove attribute with invalid key.")
             return
        key_stripped = key.strip()
        if key_stripped in self.attributes:
            del self.attributes[key_stripped]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "name": self.name,
            "attributes": self.attributes.copy()
        }

    @staticmethod
    def from_dict(data: dict) -> 'KnowledgeNode':
        node = KnowledgeNode(data.get("node_type", "Unknown"), data.get("name", "Unnamed"))
        node.id = data.get("id", str(uuid.uuid4()))
        node.attributes = data.get("attributes", {})
        return node

    def __str__(self) -> str:
        return f"KnowledgeNode(Type: '{self.node_type}', Name: '{self.name}', Attributes: {self.attributes})"

    def __repr__(self) -> str:
        return f"KnowledgeNode(node_type='{self.node_type}', name='{self.name}', attributes={self.attributes}, id='{self.id}')"


# --- Avra Core Component: Knowledge Relationship ---
class KnowledgeRelationship:
    """Represents a directed relationship (edge) between two KnowledgeNodes."""
    def __init__(self, source_node_id: str, target_node_id: str, rel_type: str, attributes: dict = None):
        if not isinstance(source_node_id, str) or not source_node_id.strip():
             raise ValueError("source_node_id must be a non-empty string.")
        if not isinstance(target_node_id, str) or not target_node_id.strip():
             raise ValueError("target_node_id must be a non-empty string.")
        if not isinstance(rel_type, str) or not rel_type.strip():
             raise ValueError("rel_type must be a non-empty string.")
        if attributes is not None and not isinstance(attributes, dict):
             raise ValueError("attributes must be a dictionary or None.")

        self.id: str = str(uuid.uuid4())
        self.source_id: str = source_node_id.strip()
        self.target_id: str = target_node_id.strip()
        self.rel_type: str = rel_type.strip()
        self.attributes: dict = attributes if attributes is not None else {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "rel_type": self.rel_type,
            "attributes": self.attributes.copy()
        }

    @staticmethod
    def from_dict(data: dict) -> 'KnowledgeRelationship':
        rel = KnowledgeRelationship(
            data.get("source_id", ""),
            data.get("target_id", ""),
            data.get("rel_type", "unknown")
        )
        rel.id = data.get("id", str(uuid.uuid4()))
        rel.attributes = data.get("attributes", {})
        return rel

    def __str__(self) -> str:
        return f"KnowledgeRelationship(ID: {self.id}, Type: '{self.rel_type}', SourceID: {self.source_id} -> TargetID: {self.target_id}, Attributes: {self.attributes})"

    def __repr__(self) -> str:
        return f"KnowledgeRelationship(source_id='{self.source_id}', target_id='{self.target_id}', rel_type='{self.rel_type}', attributes={self.attributes}, id='{self.id}')"


# --- Knowledge Graph Structure (In-Memory) ---
class AvraKnowledgeGraph:
    def __init__(self):
        self.nodes: dict[str, KnowledgeNode] = {} # Keyed by ID
        self._nodes_by_name: dict[str, str] = {} # Secondary index: lowercased name -> ID
        self.relationships: dict[str, KnowledgeRelationship] = {} # Keyed by ID
        # Indices for faster traversal (ID -> List of Relationships)
        self._relationships_by_source: dict[str, list[str]] = {} # Source Node ID -> List of Relationship IDs
        self._relationships_by_target: dict[str, list[str]] = {} # Target Node ID -> List of Relationship IDs

    def add_node(self, node: KnowledgeNode):
        if node.id in self.nodes:
            pass

        self.nodes[node.id] = node
        self._nodes_by_name[node.name.strip().lower()] = node.id # Add/update name index

    def get_node_by_id(self, node_id: str) -> KnowledgeNode | None:
        return self.nodes.get(node_id)

    def find_node_by_name(self, name: str) -> KnowledgeNode | None:
        node_id = self._nodes_by_name.get(name.strip().lower())
        if node_id:
            return self.get_node_by_id(node_id)
        return None

    def add_relationship(self, rel: KnowledgeRelationship):
        if rel.id in self.relationships:
            pass

        if rel.source_id not in self.nodes or rel.target_id not in self.nodes:
            return

        self.relationships[rel.id] = rel

        self._relationships_by_source.setdefault(rel.source_id, []).append(rel.id)
        self._relationships_by_target.setdefault(rel.target_id, []).append(rel.id)


    def get_relationship_by_id(self, rel_id: str) -> KnowledgeRelationship | None:
        return self.relationships.get(rel_id)

    def get_relationships_from(self, source_node_id: str) -> list[KnowledgeRelationship]:
        rel_ids = self._relationships_by_source.get(source_node_id, [])
        return [self.relationships[rel_id] for rel_id in rel_ids if rel_id in self.relationships]

    def get_relationships_to(self, target_node_id: str) -> list[KnowledgeRelationship]:
        rel_ids = self._relationships_by_target.get(target_node_id, [])
        return [self.relationships[rel_id] for rel_id in rel_ids if rel_id in self.relationships]

    def get_connected_nodes(self, node_id: str, rel_type: str = None, direction: str = "both") -> list[KnowledgeNode]:
        """Basic graph query: Find nodes connected to a given node (1 hop)."""
        connected_nodes: list[KnowledgeNode] = []
        if node_id not in self.nodes:
            return connected_nodes

        if direction in ["from", "both"]:
            for rel in self.get_relationships_from(node_id):
                if rel_type is None or rel.rel_type.lower() == rel_type.lower():
                    target_node = self.get_node_by_id(rel.target_id)
                    if target_node and target_node.id != node_id and target_node not in connected_nodes:
                        connected_nodes.append(target_node)

        if direction in ["to", "both"]:
            for rel in self.get_relationships_to(node_id):
                if rel_type is None or rel.rel_type.lower() == rel_type.lower():
                    source_node = self.get_node_by_id(rel.source_id)
                    if source_node and source_node.id != node_id and source_node not in connected_nodes:
                        connected_nodes.append(source_node)

        return connected_nodes

    def find_paths(self, start_node_name: str, max_hops: int = 2) -> list[list[KnowledgeNode]]:
        """Graph query: Find all paths starting from a node name up to max_hops (BFS)."""
        start_node = self.find_node_by_name(start_node_name)
        if not start_node:
            return []

        paths: list[list[KnowledgeNode]] = []
        queue = [(start_node, [start_node])]

        while queue:
            current_node, path = queue.pop(0)
            paths.append(path)

            if len(path) >= max_hops + 1:
                continue

            neighbors = self.get_connected_nodes(current_node.id)
            for neighbor_node in neighbors:
                 if neighbor_node.id not in [n.id for n in path]:
                      new_path = path + [neighbor_node]
                      queue.append((neighbor_node, new_path))

        return [p for p in paths if len(p) > 1 or max_hops == 0]


    def to_dict(self) -> dict:
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "relationships": [rel.to_dict() for rel in self.relationships.values()]
        }

    @staticmethod
    def from_dict(data: dict) -> 'AvraKnowledgeGraph':
        graph = AvraKnowledgeGraph()
        if "nodes" in data and isinstance(data["nodes"], list):
            for node_data in data["nodes"]:
                try:
                    node = KnowledgeNode.from_dict(node_data)
                    graph.add_node(node)
                except Exception as e:
                    print(f"Warning: Failed to load node data: {node_data}. Error: {e}")

        if "relationships" in data and isinstance(data["relationships"], list):
             for rel_data in data["relationships"]:
                 try:
                     rel = KnowledgeRelationship.from_dict(rel_data)
                     graph.add_relationship(rel)
                 except Exception as e:
                     print(f"Warning: Failed to load relationship data: {rel_data}. Error: {e}")

        return graph


# --- Persistence Functions (For AvraKnowledgeGraph object) ---
def save_knowledge_graph(graph: AvraKnowledgeGraph, filename: str = MEMORY_FILE):
    """Saves the entire knowledge graph to a JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(graph.to_dict(), f, indent=4)
    except Exception as e:
        print(f"\nError saving Knowledge Graph to {filename}: {e}")

def load_knowledge_graph(filename: str = MEMORY_FILE) -> AvraKnowledgeGraph:
    """Loads the knowledge graph from a JSON file."""
    graph = AvraKnowledgeGraph()
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            graph = AvraKnowledgeGraph.from_dict(data)
            print(f"\n--- Knowledge Graph loaded from {filename} ---")
        except json.JSONDecodeError:
             print(f"\nWarning: Could not decode JSON from {filename}. Starting with empty graph.")
        except Exception as e:
            print(f"\nError loading Knowledge Graph from {filename}: {e}. Starting with empty graph.")
    else:
        print(f"\n--- Knowledge Graph file {filename} not found. Starting with empty graph. ---")
    return graph

# --- Global Knowledge Graph Instance ---
avra_knowledge_graph: AvraKnowledgeGraph = AvraKnowledgeGraph()


# --- Helper for JSON Extraction ---
def extract_json_from_text(text: str) -> dict | None:
    """
    Attempts to find and return a JSON object string containing 'nodes' and 'relationships',
    handling markdown code blocks (```json ... ```) and annotations.
    """
    # Regex to find a JSON object, possibly within a ```json code block
    match = re.search(r"```json\s*(\{.*?\"nodes\".*?\"relationships\".*?\})\s*```", text, re.DOTALL)
    if match:
        json_content = match.group(1)
        json_content = re.sub(r"cref\{.*?\}", "", json_content)
        json_content = re.sub(r'\S*?"\s*"?name"\s*:', '"name":', json_content)
        json_content = re.sub(r'\S*?"\s*"?source"\s*:', '"source":', json_content)
        json_content = re.sub(r'\S*?"\s*"?target"\s*:', '"target":', json_content)
        json_content = re.sub(r'\S*?"\s*"?type"\s*:', '"type":', json_content)

        try:
            parsed_json = json.loads(json_content)
            if isinstance(parsed_json, dict) and 'nodes' in parsed_json and 'relationships' in parsed_json:
                 return parsed_json
            return None
        except json.JSONDecodeError:
            return None
        except Exception as e:
             return None


    # If not found in markdown, try to find a standalone JSON object with nodes/relationships
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_content = text[first_brace : last_brace + 1]
        json_content = re.sub(r"cref\{.*?\}", "", json_content)
        json_content = re.sub(r'\S*?"\s*"?name"\s*:', '"name":', json_content)
        json_content = re.sub(r'\S*?"\s*"?source"\s*:', '"source":', json_content)
        json_content = re.sub(r'\S*?"\s*"?target"\s*:', '"target":', json_content)
        json_content = re.sub(r'\S*?"\s*"?type"\s*:', '"type":', json_content)

        try:
            parsed_json = json.loads(json_content)
            if isinstance(parsed_json, dict) and 'nodes' in parsed_json and 'relationships' in parsed_json:
                 return parsed_json
            return None
        except json.JSONDecodeError:
            return None
        except Exception as e:
             return None

    return None


# --- Conceptual Neuron: Input/Extraction ---
# Returns a structured result indicating success/failure and the data
def extraction_neuron(user_input: str, gemini_client: genai.Client, model_name: str, retry_count: int = 0) -> dict:
     """
     Conceptual Extraction Neuron: Uses Gemini to identify entities/concepts (nodes)
     and connections between them (relationships) from input.
     Returns: {'status': 'SUCCESS', 'data': {...}} or {'status': 'FAILED', 'error': '...'}
     """
     extraction_prompt = textwrap.dedent(f"""
     You are an entity and relationship extraction specialist. Analyze the following user input and identify key entities or concepts (nodes) and connections between them (relationships).
     For each identified node, suggest a likely node type (e.g., Person, Object, Concept, Task, Location, Animal, Tool, Attribute).
     For each identified relationship, specify the source node (by its name), the relationship type (e.g., has_name, is_a, lives_in, owns, has_job, has_attribute, married_to, acquired_from, is_daughter_of, wants_to_travel_to, plans_action_on, wants_to_return_to, has_city, wants_to_ditch, has_sister, inquires_about, possesses, has_state, lacks_knowledge_about), and the target node (by its name or value if it's a literal). Ensure relationship types are consistent where possible.
     Format the output as a single JSON object with two keys: "nodes" (a list of node objects like {{"name": "Entity Name", "node_type": "EntityType"}}) and "relationships" (a list of relationship objects like {{"source": "SourceName", "type": "RelationshipType", "target": "TargetName"}}).
     Output ONLY the JSON object, nothing else, unless enclosing in ```json for clarity is necessary. Avoid adding any other annotations or text outside the JSON.

     User Input: "{user_input}"
     """)

     print(f"\n--- Extraction Neuron activated (Attempt {retry_count + 1}) ---")

     extraction_config = types.GenerateContentConfig(
         response_mime_type="text/plain",
     )

     try:
         extraction_response = gemini_client.models.generate_content(
             model=model_name,
             contents=[types.Content(role="user", parts=[types.Part.from_text(text=extraction_prompt)])],
             config=extraction_config,
         )
         extracted_raw_text = extraction_response.text.strip()

         parsed_extraction = extract_json_from_text(extracted_raw_text)

         if parsed_extraction:
             print("  Extraction Neuron: SUCCESS - Data parsed.")
             # print(f"  Parsed Data: {parsed_extraction}")
             return {'status': 'SUCCESS', 'data': parsed_extraction}
         else:
              print("  Extraction Neuron: FAILED - Parsing failed.")
              # print(f"  Raw extraction response was: {extracted_raw_text}")
              return {'status': 'FAILED', 'error': 'Parsing failed', 'raw_output': extracted_raw_text}

     except Exception as e:
         print(f"  Extraction Neuron: FAILED - API Error: {e}")
         return {'status': 'FAILED', 'error': f'API Error: {e}'}


# --- Conceptual Neuron: Memory Management ---
# Returns a structured result indicating success/failure and the memory context data
def memory_management_neuron(extracted_data: dict | None, user_input: str) -> dict:
    """
    Conceptual Memory Management Neuron: Interacts with the Knowledge Graph.
    Adds nodes and relationships based on extracted data.
    Performs graph queries and returns a dictionary summarizing memory interactions and retrieved info,
    and signals if significant context was found.
    Returns: {'status': 'SUCCESS', 'context_summary': list[str], 'found_context': bool}
             or {'status': 'FAILED', 'error': '...'}
    """
    global avra_knowledge_graph

    print("\n--- Memory Management Neuron activated ---")

    context_summary_strings = []
    queried_context_strings = set()

    nodes_added_or_found_by_name = {} # Track nodes by name for relationship linking

    found_meaningful_context = False

    if extracted_data and isinstance(extracted_data.get('nodes'), list):
        for node_data in extracted_data['nodes']:
            if isinstance(node_data, dict) and 'name' in node_data and 'node_type' in node_data:
                node_name = node_data.get('name', '').strip()
                node_type = node_data.get('node_type', '').strip()

                if not node_name or not node_type:
                     continue

                existing_node = avra_knowledge_graph.find_node_by_name(node_name)

                if existing_node:
                    print(f"  Memory Neuron: Found existing node: {existing_node.name} (ID: {existing_node.id})")
                    nodes_added_or_found_by_name[node_name.lower()] = existing_node.id
                else:
                    new_node = KnowledgeNode(node_type=node_type, name=node_name)
                    new_node.add_attribute("source_input", user_input[:100] + ("..." if len(user_input) > 100 else ""))
                    avra_knowledge_graph.add_node(new_node)
                    print(f"  Memory Neuron: Added new node: {new_node.name} (ID: {new_node.id})")
                    nodes_added_or_found_by_name[node_name.lower()] = new_node.id
                    context_summary_strings.append(f"Added new node: {new_node.name} (Type: {new_node.node_type})")


        if isinstance(extracted_data.get('relationships'), list):
            for rel_data in extracted_data['relationships']:
                if isinstance(rel_data, dict) and 'source' in rel_data and 'type' in rel_data and 'target' in rel_data:
                    source_name = rel_data.get('source', '').strip()
                    rel_type = rel_data.get('type', '').strip()
                    target_name = rel_data.get('target', '').strip()

                    if not source_name or not rel_type or not target_name:
                        continue

                    source_id = nodes_added_or_found_by_name.get(source_name.lower())
                    target_id = nodes_added_or_found_by_name.get(target_name.lower())

                    if source_id and target_id:
                        existing_rels = avra_knowledge_graph.get_relationships_from(source_id)
                        already_exists = False
                        for existing_rel in existing_rels:
                            # Basic check for duplicate relationship (same source, target, type)
                            if existing_rel.target_id == target_id and existing_rel.rel_type.lower() == rel_type.lower():
                                already_exists = True
                                break

                        if not already_exists:
                            new_rel = KnowledgeRelationship(source_id, target_id, rel_type)
                            avra_knowledge_graph.add_relationship(new_rel)
                            print(f"  Memory Neuron: Added new relationship: '{source_name}' --('{rel_type}')--> '{target_name}'")
                            context_summary_strings.append(f"Added relationship: '{source_name}' --('{rel_type}')--> '{target_name}'")
                        else:
                             print(f"  Memory Neuron: Relationship already exists: '{source_name}' --('{rel_type}')--> '{target_name}'")

                    else:
                        print(f"Warning: Memory Neuron: Skipping relationship '{source_name}' --('{rel_type}')--> '{target_name}' due to missing required nodes in current extraction or graph.")

    # --- Perform Graph Queries to Enrich Context ---
    relevant_node_ids = [avra_knowledge_graph._nodes_by_name.get(name) for name in nodes_added_or_found_by_name.keys() if avra_knowledge_graph._nodes_by_name.get(name)]

    for node_id in relevant_node_ids:
        if node_id:
             node = avra_knowledge_graph.get_node_by_id(node_id)
             if node:
                 # Add attributes
                 if node.attributes:
                     attrs_str = ", ".join([f"{k}: {v}" for k, v in node.attributes.items()])
                     queried_context_strings.add(f"- Details for '{node.name}' (Type: {node.node_type}): Attributes: {{{attrs_str}}}")
                     found_meaningful_context = True

                 # Find and add 1-hop related info
                 for rel in avra_knowledge_graph.get_relationships_from(node.id) + avra_knowledge_graph.get_relationships_to(node.id):
                      related_node_id = rel.target_id if rel.source_id == node.id else rel.source_id
                      related_node = avra_knowledge_graph.get_node_by_id(related_node_id)
                      if related_node and related_node.id != node.id:
                           queried_context_strings.add(f"- Related to '{node.name}' (1 hop): '{rel.rel_type}' with '{related_node.name}' (Type: {related_node.node_type})")
                           found_meaningful_context = True
                           if related_node.attributes:
                                related_attrs_str = ", ".join([f"{k}: {v}" for k, v in related_node.attributes.items()])
                                queried_context_strings.add(f"- Details for '{related_node.name}' (Type: {related_node.node_type}): Attributes: {{{related_attrs_str}}}")


                 # --- Multi-hop query example (up to 2 hops) ---
                 paths = avra_knowledge_graph.find_paths(node.name, max_hops=2)

                 if paths:
                     for path in paths:
                          if len(path) > 1:
                               path_desc = " -> ".join([n.name for n in path])
                               start_node_path = path[0]
                               end_node_path = path[-1]
                               queried_context_strings.add(f"- Path from '{start_node_path.name}' ({len(path)-1} hops): {path_desc}")
                               found_meaningful_context = True

                               if end_node_path.id != node.id and end_node_path.attributes:
                                    end_attrs_str = ", ".join([f"{k}: {v}" for k, v in end_node_path.attributes.items()])
                                    queried_context_strings.add(f"- Details for Path End '{end_node_path.name}' (Type: {end_node_path.node_type}): Attributes: {{{end_attrs_str}}}")


    final_context_strings = context_summary_strings + list(queried_context_strings)

    if not extracted_data:
         print("  Memory Management Neuron: Skipped due to failed extraction.")
         return {'status': 'FAILED', 'error': 'No extracted data'}

    return {'status': 'SUCCESS', 'context_summary': final_context_strings, 'found_context': found_meaningful_context}


# --- Tool Execution Function ---
def execute_python_code(code_string: str) -> dict:
    """
    Safely executes Python code in a limited environment and captures output/errors.
    Returns: {'outcome': 'OK', 'output': '...'} or {'outcome': 'ERROR', 'output': '...'}
    """
    print("\n[Executing Code...]")
    # Redirect stdout and stderr to capture output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = io.StringIO()
    redirected_error = io.StringIO()
    sys.stdout = redirected_output
    sys.stderr = redirected_error

    execution_output = ""
    outcome = types.CodeExecutionResult.Outcome.OUTCOME_ERROR # Assume error until success

    # Define a limited execution environment
    # Only allow a subset of built-in functions and modules
    allowed_builtins = {
        'print': print,
        'len': len,
        'str': str,
        'int': int,
        'float': float,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'set': set,
        'sum': sum,
        'max': max,
        'min': min,
        'abs': abs,
        'round': round,
        'bool': bool,
        'True': True,
        'False': False,
        'None': None,
    }
    allowed_modules = ['math', 'json', 're'] # Allow math, json, regex for typical AI tasks

    # Prepare the execution environment
    execution_globals = {"__builtins__": allowed_builtins}
    for module_name in allowed_modules:
        try:
            execution_globals[module_name] = __import__(module_name)
        except ImportError:
            pass # Module not available, skip

    try:
        # Execute the code
        exec(code_string, execution_globals)

        # Capture output
        execution_output = redirected_output.getvalue().strip()
        error_output = redirected_error.getvalue().strip()

        if error_output:
            outcome = types.CodeExecutionResult.Outcome.OUTCOME_ERROR
            execution_output = error_output # Report error message as output
        else:
            outcome = types.CodeExecutionResult.Outcome.OUTCOME_OK
            if not execution_output: # If no explicit print, maybe a calculation result implicitly?
                 # This is tricky. For now, if OK but no output, report that.
                 execution_output = "Execution successful, no output."


    except Exception as e:
        # Capture execution errors
        outcome = types.CodeExecutionResult.Outcome.OUTCOME_ERROR
        execution_output = f"Execution Error: {e}"

    finally:
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        print("[...Execution Complete]")

    return {'outcome': outcome, 'output': execution_output}


# --- Conceptual Neuron: Response Generation ---
# Returns a structured result
def response_neuron(chat_session: genai.ChatSession, user_input: str, memory_context_data: dict) -> dict:
    """
    Conceptual Response Generation Neuron: Synthesizes the final output using a Chat Session.
    Handles processing tool requests and sends results back to the session.
    Returns: {'status': 'SUCCESS', 'response': '...'} or {'status': 'FAILED', 'error': '...'}
    """
    memory_context_strings = memory_context_data.get('context_summary', [])
    found_meaningful_context = memory_context_data.get('found_context', False)

    memory_context_string = "\n".join(memory_context_strings) if memory_context_strings else "Memory Context: No relevant information found or added in this turn."

    prompt_instruction = "Use the information from the provided 'Memory Context' which includes details about entities, relationships, and retrieved information from your knowledge graph, to inform your answer to the 'User Input'. If the Memory Context provides a direct answer or relevant facts, prioritize using that information."
    if not found_meaningful_context:
        prompt_instruction = "No significant relevant information was found in your knowledge memory for the current input. Generate a helpful and relevant response based primarily on the user input, but keep the context format in mind for future turns. If new information is mentioned, you know it will be added to memory."


    prompt_content_parts = [
        types.Part.from_text(text=textwrap.dedent(f"""
        You are Avra, an AI assistant. Your task is to generate a helpful and relevant response to the user, demonstrating awareness of your knowledge memory and ability to use tools.
        {prompt_instruction}
        If you need to perform a calculation, process data, or use a programming function to answer, use the code execution tool.
        If new information is mentioned in the user input that wasn't specifically listed in the Memory Context, try to incorporate it naturally into your response if relevant.
        Be concise and helpful.

        Memory Context:
        {memory_context_string}

        User Input: "{user_input}"

        Avra's Response:
        """))
    ]

    # --- Define the Code Execution Tool for the Chat Session ---
    available_tools = [
        types.Tool(code_execution=types.ToolCodeExecution),
    ]

    # Configuration for messages sent to the chat session
    chat_config = types.GenerateContentConfig(
        tools=available_tools,
        # response_mime_type="text/plain", # Usually not set explicitly for chat
    )

    print("\n--- Response Neuron sending message to Chat Session ---")

    full_response_text = ""
    try:
        # Send the combined prompt (user input + memory context) to the chat session
        # The ChatSession manages history and handles basic tool request/response turn sequences
        response = chat_session.send_message(
            prompt_content_parts,
            # stream=True, # Streaming adds complexity with tool use in this loop
            # config=chat_config # Config usually set on chat.create, but can override here
        )

        # Process the response from the chat session
        # This response might contain text OR a tool request
        candidate = response.candidates[0] if response.candidates else None
        if not candidate or not candidate.content or not candidate.content.parts:
             print("  Response Neuron: Received empty response.")
             return {'status': 'FAILED', 'error': 'Empty response from model'}

        # Check if the model is asking to execute code
        if candidate.content.parts[0].executable_code:
             code_to_execute = candidate.content.parts[0].executable_code.code
             print(f"  Response Neuron received Code Request: {code_to_execute}")

             # --- Conceptual Neuron: Tool Execution (Orchestrator handles this delegation) ---
             # The orchestrator captures this code request and calls the execution function.
             # The result is then sent back to the chat session by the orchestrator.

             return {'status': 'TOOL_REQUESTED', 'code': code_to_execute} # Signal orchestrator

        # Check if the model returned a text response
        if candidate.content.parts[0].text:
            full_response_text = candidate.content.parts[0].text
            print(f"Avra: {full_response_text}") # Print the text response
            return {'status': 'SUCCESS', 'response': full_response_text}

        # Handle other potential part types or unexpected content
        print(f"  Response Neuron: Received unexpected content part(s).")
        return {'status': 'FAILED', 'error': 'Received unexpected content part(s)'}


    except Exception as e:
        print(f"\nError during response generation with Gemini API: {e}")
        return {'status': 'FAILED', 'error': f'API Error: {e}'}


# --- Main Execution Loop (Orchestrator/Central Neuron) ---
if __name__ == "__main__":
    print("--- Avra AGI (at0) Colab Memory Integration Demo ---")
    print("Type 'quit', 'exit', or 'q' to end the session.")

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please ensure your Colab setup cell correctly retrieves the GOOGLE_API_KEY secret and sets the GEMINI_API_KEY environment variable.")
        print("\nSetup cell example:")
        print("!pip install -U -q \"google.genai\"")
        print("import os")
        from google.colab import userdata
        print("os.environ[\"GEMINI_API_KEY\"] = userdata.get(\"GOOGLE_API_KEY\")")
        exit()


    # --- Initialize Gemini Client ---
    print(f"\nAttempting to use Gemini model: {MODEL_NAME_TO_USE}")
    try:
        gemini_client = genai.Client(api_key=api_key)
        print("Gemini API Client initialized.")

    except Exception as e:
        print(f"Error initializing Gemini API Client: {e}")
        exit()

    # --- Load Knowledge Graph on Startup (Orchestrator) ---
    avra_knowledge_graph = load_knowledge_graph(MEMORY_FILE)

    # --- Initialize Chat Session (Orchestrator/Central Neuron) ---
    # Create a new chat session for each user session to maintain history
    print("\n--- Starting new Chat Session ---")
    # Configuration for the chat session (includes tools)
    chat_session_config = types.GenerateContentConfig(
         tools=[types.Tool(code_execution=types.ToolCodeExecution)],
    )
    # Initial message to prime the model? Or just create the session.
    # For now, just create with tools config.
    chat_session = gemini_client.chats.create(
        model=MODEL_NAME_TO_USE,
        config=chat_session_config,
    )
    print("--- Chat Session created ---")


    # --- Chat Loop (Orchestrator/Central Neuron) ---
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        if not user_input.strip():
            continue # Skip empty input

        print(f"\n--- Orchestrator processing input: {user_input} ---")

        # --- Orchestration Step 1: Delegate to Extraction Neuron with Retries ---
        extracted_data_result = {'status': 'FAILED'}
        for i in range(MAX_EXTRACTION_RETRIES + 1):
             extracted_data_result = extraction_neuron(user_input, gemini_client, MODEL_NAME_TO_USE, retry_count=i)
             if extracted_data_result['status'] == 'SUCCESS':
                  break

        if extracted_data_result['status'] != 'SUCCESS':
            print("\n--- Orchestrator: Extraction failed after multiple retries. Cannot fully update memory. ---")
            # Still attempt memory management with whatever (potentially None) data was extracted on last try
            # and proceed to response with limited info.
            memory_management_result = memory_management_neuron(extracted_data_result.get('data'), user_input)
            save_knowledge_graph(avra_knowledge_graph, MEMORY_FILE)
            response_neuron(chat_session, user_input, memory_management_result)
            continue # Go to next user input


        # --- Orchestration Step 2: Delegate to Memory Management Neuron ---
        memory_management_result = memory_management_neuron(extracted_data_result['data'], user_input)

        if memory_management_result['status'] != 'SUCCESS':
             print("\n--- Orchestrator: Memory management failed. Cannot proceed with response based on updated memory. ---")
             # Still attempt response with whatever memory context was generated before failure
             save_knowledge_graph(avra_knowledge_graph, MEMORY_FILE)
             response_neuron(chat_session, user_input, memory_management_result)
             continue # Go to next user input


        # --- Orchestration Step 3: Save Graph (Orchestrator task) ---
        save_knowledge_graph(avra_knowledge_graph, MEMORY_FILE)


        # --- Orchestration Step 4: Delegate to Response Generation Neuron (Initial Call) ---
        # This sends the user input + memory context to the chat session
        response_result = response_neuron(chat_session, user_input, memory_management_result)

        # --- Orchestration Step 5: Handle Tool Requests and Feedback Loop ---
        # If the response neuron signalled a tool request, execute the code and send the result back
        tool_execution_count = 0
        while response_result['status'] == 'TOOL_REQUESTED' and tool_execution_count <= MAX_TOOL_EXECUTION_RETRIES:
            code_to_execute = response_result['code']
            tool_execution_count += 1
            print(f"\n--- Orchestrator executing Tool Request (Attempt {tool_execution_count}) ---")

            # Execute the code
            execution_result_data = execute_python_code(code_to_execute)

            # Send the result back to the chat session
            print("\n--- Orchestrator sending Tool Result back to Chat Session ---")
            try:
                 # The format for sending tool results back is specific
                 tool_result_content = types.Content(
                     role='tool', # Role is 'tool' for results
                     parts=[types.Part.from_code_execution_result(types.CodeExecutionResult(
                         outcome=execution_result_data['outcome'],
                         execution_output=execution_result_data['output'] # Use execution_output here for sending results
                     ))]
                 )
                 # Send the tool result message to the chat session
                 response_result = response_neuron(chat_session, tool_result_content, memory_management_result) # Pass the result back to the neuron/chat

            except Exception as e:
                 print(f"\n--- Orchestrator Error sending Tool Result back to Chat: {e} ---")
                 response_result = {'status': 'FAILED', 'error': f'Error sending tool result: {e}'}
                 break # Exit tool loop if sending fails

            if response_result['status'] != 'TOOL_REQUESTED':
                 # If the model didn't request another tool call, the loop ends.
                 break
            # If it requested another tool call, the loop continues (up to MAX_TOOL_EXECUTION_RETRIES)


        # --- Orchestration Step 6: Final Response ---
        if response_result['status'] == 'SUCCESS':
             # Final text response was already printed by the response_neuron
             pass
        elif response_result['status'] == 'TOOL_REQUESTED':
             print("\n--- Orchestrator: Tool execution loop ended without a final text response. Max retries reached or unexpected state. ---")
             # Orchestrator could generate a fallback response here
             print("Avra: I processed the tool request but could not generate a final text response.")
             print("---------------------------------------")
        else:
             print("\n--- Orchestrator: An error occurred during the response process. ---")
             # Error message was already printed by the response_neuron


    print("\n--- Avra Session Ended ---")