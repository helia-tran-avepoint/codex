import base64
import importlib
import json
import os
import threading
import time
import zipfile
from datetime import datetime
from operator import index
from typing import cast

import graphviz
import requests
import streamlit as st
import streamlit.components.v1 as components
from llm_proxy import process_message, send_message
from streamlit.components.v1 import html
from streamlit_agraph import Config, Edge, Node, agraph
from streamlit_javascript import st_javascript

from shared.models import (
    PROJECT_CONFIGS_PATH,
    PrivateKnowledge,
    Project,
    load_project_names,
)
from webui import app_config, logger
from webui.nebula_service import connect_to_nebula, get_graph_from_nebula

from shared import constants  # isort:skip

webicon = "favicon.ico"


MAX_HISTORY = app_config.max_history

st.set_page_config(layout="wide", page_icon=webicon, page_title="AvePoint CopilotX")
window_width = st_javascript("window.innerWidth") or 1200


# @st.cache_data
def load_project(name):
    logger.debug(f"Load project {name}")
    project = Project.load(name)
    return project


def build_index(knowledge, project_name):
    logger.debug(
        f"Build project {project_name}, knowledge {knowledge.knowledge_name} index"
    )
    data = {
        "path": knowledge.knowledge_dir,
        "data_type": knowledge.knowledge_type,
        "project_name": project_name,
        "knowledge_name": knowledge.knowledge_name,
    }

    response = requests.post(
        f"http://{constants.INDEX_SERVICE_URL}:{app_config.index_service_port}/build_index",
        json=data,
    )
    if response.status_code == 200:
        st.cache_data.clear()
        st.rerun()
        st.success(f"Index built for {knowledge.knowledge_name}")
    else:
        st.error(f"Failed to build index for {knowledge.knowledge_name}")


def build_index_async(knowledge, project_name):
    threading.Thread(target=build_index, args=(knowledge, project_name)).start()


def check_name_unique():
    pass


if "knowledges_indexed" not in st.session_state:
    st.session_state.knowledge_indexed = False

if "load_project_index" not in st.session_state:
    st.session_state.load_project_index = False

with st.sidebar:
    project_names = load_project_names()
    selected_project = st.selectbox("Select Project", project_names)

    if selected_project:
        project = load_project(selected_project)

        # limit only can add 2 knowledge, which should include both document type and source code type. for now only focus on TPS project.
        st.session_state.knowledge_indexed = (
            all(knowledge.indexed for knowledge in project.knowledges)
            and len(project.knowledges) == 2
            and any(
                knowledge.knowledge_type == "document"
                for knowledge in project.knowledges
            )
            and any(
                knowledge.knowledge_type == "source_code"
                for knowledge in project.knowledges
            )
        )

        st.markdown("### Current Knowledge")
        for i, knowledge in enumerate(project.knowledges):
            st.markdown(f"**Knowledge {knowledge.knowledge_name}**")
            # st.text(f"Directory: {knowledge.knowledge_dir}")
            st.text(f"Type: {knowledge.knowledge_type}")
            st.text(f"Indexed: {knowledge.indexed}")
            if knowledge.indexed:
                st.text(f"Last Build Time: {knowledge.last_index_time}")
            if st.button("Build Index", key=f"build_index_{i}"):
                build_index_async(knowledge, selected_project)
                st.info(f"Building index for {knowledge.knowledge_name}...")
            if st.button("Remove", key=f"remove_knowledge_{i}"):
                project.knowledges.pop(i)
                project.save()
                st.rerun()

        st.markdown("---")
        st.subheader("Add Knowledge")
        all_knowledge_name = [k.knowledge_name for k in project.knowledges]
        knowledge_name = st.text_input("Knowledge Name:", key="knowledge_name")
        # knowledge_dir = st.text_input("Knowledge Directory:", key="knowledge_dir")
        knowledge_type = st.selectbox(
            "Knowledge Type:", ["document", "source_code"], key="knowledge_type"
        )

        st.markdown("Select Knowledge Path")

        upload_status = st.empty()
        # upload_api_url = f"http://{constants.INDEX_SERVICE_URL}:{app_config.index_service_port}/upload_files"
        upload_api_url = (
            f"http://10.1.71.1:{app_config.index_service_port}/upload_files"
        )

        html_code = f"""
        <html>
            <body>
                <input type="file" id="folderUpload" webkitdirectory directory multiple>
                <script>
                    async function uploadFiles(files) {{
                        const formData = new FormData();
                        for (const file of files) {{
                            formData.append('files', file);
                            formData.append('relative_paths', file.webkitRelativePath);
                        }}
                        formData.append('project_name', "{project.name}");
                        formData.append('knowledge_name', "{knowledge_name}");
                        formData.append('knowledge_type', "{knowledge_type}");
                        try {{
                            const response = await fetch("{upload_api_url}", {{
                                method: 'POST',
                                body: formData,
                            }});
                            const result = await response.json();
                            if (response.ok) {{
                                alert('Files uploaded successfully!');
                                window.parent.postMessage("upload_complete", "*");
                            }} else {{
                                alert('Upload failed: ' + result.error);
                            }}
                        }} catch (err) {{
                            console.error('Error uploading files:', err);
                        }}
                    }}

                    document.getElementById('folderUpload').addEventListener('change', function(event) {{
                        const files = [...event.target.files];
                        uploadFiles(files);
                    }});
                </script>
            </body>
        </html>
        """
        html(html_code, height=200)

        if "upload_complete" not in st.session_state:
            st.session_state["upload_complete"] = False

        if st.session_state["upload_complete"]:
            upload_status.success(
                "Files uploaded successfully! Proceed to create Knowledge."
            )
        else:
            upload_status.info("Please upload files to proceed.")

        if st.button("Add Knowledge"):
            if knowledge_name:
                knowledge = PrivateKnowledge(
                    knowledge_dir=os.path.join(
                        app_config.shared_path,
                        "project_datas",
                        project.name,
                        knowledge_type,
                        knowledge_name,
                    ),
                    knowledge_name=knowledge_name,
                    knowledge_type=knowledge_type,
                    indexed=False,
                    last_index_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    index_dir="",
                )
                project.knowledges.append(knowledge)
                project.save()
                st.rerun()
            else:
                st.sidebar.error("Please provide a knowledge name and upload files.")

    st.markdown("---")
    st.subheader("Create New Project")
    new_project_name = st.text_input("New Project Name:", key="new_project_name")
    if st.button("Create Project"):
        new_project = Project(new_project_name, [])
        new_project.save()
        st.rerun()

    st.markdown("---")
    st.subheader("Evaluation")
    if st.button("Evaluation Dashboard"):
        response = requests.get(
            f"http://{constants.INDEX_SERVICE_URL}:{app_config.index_service_port}/run-dashboard",
        )
        st.sidebar.info(response.json()["message"])

    st.markdown("---")
    st.subheader("Load Porject Index")
    if st.button("Load Index"):
        payload = {"project_name": selected_project}
        response = requests.post(
            f"http://{constants.INDEX_SERVICE_URL}:{app_config.index_service_port}/load_index",
            json=payload,
        )
        st.session_state.load_project_index = True

st.title("💬 CopilotX Assistant")

# st.subheader("Method Call Chain")

# connection_pool = connect_to_nebula()
# current_node = st.session_state.get("current_node", None)
# # nodes, edges = get_graph_from_nebula(connection_pool, current_node)
# initial_nodes = [
#     {"id": "UI", "label": "User Interface"},
#     {"id": "Login", "label": "Login Module"},
#     {"id": "DB", "label": "Database"},
# ]

# initial_edges = [
#     {"source": "UI", "target": "Login", "label": "calls"},
#     {"source": "Login", "target": "DB", "label": "connects_to"},
# ]

# login_expanded_nodes = [
#     {"id": "Auth", "label": "Authentication"},
#     {"id": "Session", "label": "Session Management"},
#     {"id": "InputValidation", "label": "Input Validation"},
# ]

# login_expanded_edges = [
#     {"source": "Login", "target": "Auth", "label": "includes"},
#     {"source": "Auth", "target": "Session", "label": "manages"},
#     {"source": "Login", "target": "InputValidation", "label": "validates"},
# ]

# if "nodes_data" not in st.session_state:
#     st.session_state["nodes_data"] = initial_nodes.copy()
# if "edges_data" not in st.session_state:
#     st.session_state["edges_data"] = initial_edges.copy()
# if "expanded_nodes" not in st.session_state:
#     st.session_state["expanded_nodes"] = set()

# nodes = [
#     Node(id=n["id"], label=n["label"], link=None)
#     for n in st.session_state["nodes_data"]
# ]
# edges = [
#     Edge(source=e["source"], target=e["target"], label=e["label"])
#     for e in st.session_state["edges_data"]
# ]

# config = Config(
#     width=window_width - 50,
#     height=700,
#     directed=True,
#     hierarchical=True,
#     physics=False,
#     groups={
#         "default": {"color": {"background": "lightblue", "border": "blue"}},
#         "expanded": {"color": {"background": "lightgreen", "border": "green"}},
#     },
# )


# agraph_result = agraph(nodes=nodes, edges=edges, config=config)

# if agraph_result:
#     selected_node = agraph_result
#     st.session_state["current_node"] = selected_node
#     st.write(f"Current selected node: {selected_node}")

#     if selected_node == "Login" and st.button("Expand Login Node"):
#         if selected_node not in st.session_state["expanded_nodes"]:
#             st.session_state["nodes_data"].extend(login_expanded_nodes)
#             st.session_state["edges_data"].extend(login_expanded_edges)
#             st.session_state["expanded_nodes"].add(selected_node)
#         st.rerun()

#     if selected_node == "Login" and st.button("Collapse Login Node"):
#         if selected_node in st.session_state["expanded_nodes"]:
#             st.session_state["nodes_data"] = [
#                 n
#                 for n in st.session_state["nodes_data"]
#                 if n["id"] not in {n["id"] for n in login_expanded_nodes}
#             ]
#             st.session_state["edges_data"] = [
#                 e
#                 for e in st.session_state["edges_data"]
#                 if (e["source"], e["target"])
#                 not in {(e["source"], e["target"]) for e in login_expanded_edges}
#             ]
#             st.session_state["expanded_nodes"].remove(selected_node)
#         st.rerun()


# html_container = st.empty()

websocket_code = """
<script type="text/javascript">
    window.onload = function() {
        console.log("WebSocket script loaded...");
        
        function clearThink() {
            window.parent.document.getElementById('think').innerHTML = "";
        }
        
        function waitForThinkElement(callback) {
            const thinkElement = window.parent.document.getElementById('think');
            if (thinkElement) {
                callback(thinkElement);
            } else {
                console.warn("Element #think not found, retrying in 100ms...");
                setTimeout(() => waitForThinkElement(callback), 100);
            }
        }

        const ws = new WebSocket('ws://10.1.71.1:web_socket_port');

        ws.onopen = function() {
            console.log("WebSocket connected.");
        };

        ws.onerror = function(error) {
            console.error("WebSocket Error:", error);
        };

        ws.onmessage = function(event) {
            console.log("Message received:", event.data);

            let message = event.data;
            message = message.replace(/\\n/g, "<br>");
            message = message.replace(/-{20,}/g, "<hr>");

            message = message.replace(/\\x1b\\[33m(.*?)\\x1b\\[0m/g, "<span style='color: #FFA500;'>$1</span>"); 
            message = message.replace(/\\x1b\\[32m(.*?)\\x1b\\[0m/g, "<span style='color: #00AA00;'>$1</span>"); 
            message = message.replace(/\\x1b\\[31m(.*?)\\x1b\\[0m/g, "<span style='color: #D32F2F; font-weight: bold;'>$1</span>"); 
            message = message.replace(/\\x1b\\[34m(.*?)\\x1b\\[0m/g, "<span style='color: #0000FF;'>$1</span>"); 
            

            message = message.replace(/CodeAnalysisTeamUserProxy \(to chat_manager\):/g, "<b style='color: red;'>📩 To Agent Manager:</b>");
            message = message.replace(/Thinking:/g, "<b style='color: blue;'>🤖 Thinking:</b>");
            message = message.replace(/Next speaker:/g, "<b style='color: #008CBA;'>📢 Speaker:</b>");
            message = message.replace(/USING AUTO REPLY.../g, "<span style='color: green; font-weight: bold;'>💻 LLM Analyzing...</span>");

            waitForThinkElement(function(thinkElement) {
                let logEntry = document.createElement("div");
                logEntry.classList.add("log-entry");
                logEntry.innerHTML = message;

                thinkElement.appendChild(logEntry);
                thinkElement.scrollTop = thinkElement.scrollHeight;
            });
        };

        ws.onclose = function() {
            console.warn("WebSocket closed. Attempting to reconnect in 5 seconds...");
            setTimeout(() => location.reload(), 5000);
        };
    };
</script>
"""

websocket_code = websocket_code.replace(
    "web_socket_port", str(app_config.web_socket_port)
)
with st.expander(" Thinking Details", icon="🧠"):
#st.dialog("Thinking Details", width="large")
#def thinkdialog():
   # with st.container(height=500):
        st.markdown(
            """
            <p><strong></strong> <span id='think'></span></p>
            """,
            unsafe_allow_html=True,
        )

#if st.button("Think", icon="🧠"):
# thinkdialog()
html(websocket_code)

# html_container.markdown(websocket_code, unsafe_allow_html=True)
# html_container.html(websocket_code)
#html(websocket_code)
st.markdown(
            """
        <style>
            .st-emotion-cache-1c7y2kd {
                flex-direction: row-reverse;
                text-align: right;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": " Hi! I’m your AI assistant!",
        }
    ]

# if current_node:
#     st.write(f"Current Node: {current_node}")
#     question = st.text_input(
#         f"Ask question for Node {current_node}",
#         placeholder="Please input your question",
#     )
#     if st.button("Send"):
#         answer = f"Your question {question} for node {current_node}"
#         st.write(f"Answer: {answer}")


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if message := st.chat_input(placeholder="Got questions? Just ask!"):
    if st.session_state.knowledge_indexed:
        # st_javascript("document.getElementById('think').innerHTML = '';")

        if (
            len(st.session_state.messages) <= 1
            and st.session_state.load_project_index == False
        ):
            if all(knowledge.indexed for knowledge in project.knowledges):
                payload = {"project_name": project.name}
                logger.debug(f"Load project {project.name} knowledge index")
                response = requests.post(
                    f"http://{constants.INDEX_SERVICE_URL}:{app_config.index_service_port}/load_index",
                    json=payload,
                )
            else:
                pass

        st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

        st.chat_message("user").write(message)
        # response = send_message(message, st.session_state.messages)
        history_messages = st.session_state.messages
        st.session_state.messages.append({"role": "user", "content": message})
        with st.spinner("Thinking in progress...", show_time=True):
            response = process_message(message, history_messages)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

        if len(st.session_state.messages) > MAX_HISTORY:
            summary_prompt = "Summarize the following conversation: " + " ".join(
                [
                    f"{msg['role']}: {msg['content']}"
                    for msg in st.session_state.messages[:-MAX_HISTORY]
                ]
            )
            summary = send_message(summary_prompt, [])
            if summary:
                st.session_state.messages = [
                    {"role": "assistant", "content": summary}
                ] + st.session_state.messages[-MAX_HISTORY:]
    else:
        st.warning(
            "Please add one document type knowledge and one source code type knowledge then build knowledge index first."
        )
