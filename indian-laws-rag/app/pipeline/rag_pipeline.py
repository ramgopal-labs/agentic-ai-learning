from app.api.models import AskRequest, AskResponse, Citation
from app.pipeline.graph import RAGGraph, RAGState


class IndianLawsRAG:
    """
    Thin adapter between the HTTP models and the LangGraph pipeline.

    All the orchestration lives in `app.pipeline.graph`; this class only translates an
    `AskRequest` into graph state and the resulting state back into an
    `AskResponse`.
    """

    def __init__(self, graph: RAGGraph | None = None):
        self.graph = graph or RAGGraph()

    def answer(self, request: AskRequest) -> AskResponse:
        initial_state: RAGState = {
            "session_id": request.session_id,
            "raw_question": request.question,
            "act_filter": request.act_filter,
            "provider": request.provider,
        }

        final_state = self.graph.invoke(initial_state)

        return AskResponse(
            answer=final_state["answer"],
            confidence=round(float(final_state.get("confidence", 0.0)), 3),
            source_path=final_state.get("source_path", "not_ready"),
            citations=[
                Citation(**citation) for citation in final_state.get("citations") or []
            ],
            rewritten_question=final_state.get("rewritten_question", request.question),
            sub_questions=final_state.get("sub_questions") or [],
        )
