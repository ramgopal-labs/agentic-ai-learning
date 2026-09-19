"""Every prompt the pipeline sends to an LLM, kept in one editable place."""

REWRITE_QUERY = (
    "Rewrite the latest user question as a standalone legal question.\n"
    "Use the conversation only to resolve references such as 'it', 'that', "
    "'the same', or 'what about'. Do not answer the question, do not add "
    "detail that the conversation does not support, and return only the "
    "rewritten question as a single line.\n\n"
    "Conversation:\n{transcript}\n\n"
    "Latest question: {question}\n\n"
    "Standalone question:"
)


DECOMPOSE_QUERY = (
    "You split compound legal questions into independent sub-questions so each "
    "can be searched separately.\n\n"
    "Rules:\n"
    "- If the question asks about exactly one thing, return it unchanged as the "
    "only item.\n"
    "- Never return more than {max_sub_queries} items.\n"
    "- Each sub-question must stand alone, without pronouns referring to the others.\n"
    '- Return a JSON array of strings and nothing else, e.g. ["...", "..."].\n\n'
    "Question: {question}\n\n"
    "JSON array:"
)


ANSWER_FROM_CONTEXT = (
    "You are an Indian-law information assistant.\n\n"
    "Rules:\n"
    "- Answer only from the supplied context. Never invent statutes, sections "
    "or facts.\n"
    "- Cite the Act and Section for every legal claim drawn from statutory "
    "context, in the form (Act name, Section N).\n"
    "- When the context comes from web sources, cite the source URL instead.\n"
    "- If the context does not answer the question, say so plainly rather than "
    "guessing.\n"
    "- Give information, not legal advice, and do not tell the user what they "
    "should do in their own case.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


NO_CONTEXT_FALLBACK = (
    "I could not find enough reliable legal context to answer this question. "
    "The local index covers only part of the Indian-Laws dataset, so try a "
    "question about an indexed Act, re-index with `--full`, or configure "
    "SERPER_API_KEY to enable the web-search fallback."
)
