from langchain_mistralai import ChatMistralAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os
from dotenv import load_dotenv

load_dotenv()

mistral_key=os.getenv("MISTRAL_API_KEY")

model=ChatMistralAI(
    api_key=mistral_key,    
    model_name="mistral-tiny",
    temperature=0.5,        
)
llm = HuggingFaceEndpoint(
    repo_id="nvidia/Nemotron-Labs-Diffusion-14B",
    task="text-generation"
)

model2 = ChatHuggingFace(
    llm=llm ,
    max_tokens=1000,
)






template1 = PromptTemplate(
    template="provide me details fact and information about {text}",
    input_variables=["text"],
)

template2 = PromptTemplate(
    template="creat a objective single choise quiz of 5 questions on {text}",
    input_variables=["text"]
)

template3 = PromptTemplate(
    template="Merge both the results and create a doucument with notres->: {topic} and quiz->:{information}",
    input_variables=["topic","information"]
)

parser=StrOutputParser()

parrallel_chain = RunnableParallel({
    "first": template1 | model | parser,
    "second": template2 | model2 | parser,
})

merge_chain= template3 | model| parser 

chain_result=parrallel_chain | merge_chain
print(chain_result)

print(parrallel_chain.get_graph().draw_ascii())



text={
    '''Rebuilding our frontend had been a long time coming because of how tricky it was to work on MDN's UI. Our previous frontend (called yari) was a React app that, unfortunately, had accumulated quite a lot of technical debt. Maintenance wasn't exactly impossible, but was certainly painful to undertake. Whenever we fixed issues or added new site functionality, we inevitably ended up piling on more technical debt. But how did we get there?

The React app had started life as a "Create React App", but a number of the built-in defaults didn't work for us. Of course, this led to a series of workarounds, and we eventually had to "eject" the configuration. We ended up with an extremely complicated Webpack config as well as some very hacky build scripts.

On the CSS side as well, things were starting to get out of control. We used Sass extensively, then added modern CSS features like CSS variables, which meant we had a bizarre mix of both idioms spread across our files.

The CSS was also incredibly entangled, with poor or nonexistent scoping. When we made a change in one UI component, we'd frequently spot unintended changes in others. These issues, and a lack of build tools to split up the CSS, meant we had to ship a large render-blocking CSS blob to our users, complete with styles for components they might never load.

But by far, the biggest issue was that our React app was merely a wrapper around our static content. To make the React app aware of the HTML content that our build tool generated would have required expensive reparsing of the HTML and an extraordinary amount of logic which we'd have to ship to users in our client-side JavaScript. We didn't want to do this, so the React app boundary essentially ended where our documentation began – we used React's dangerouslySetInnerHTML to insert the content.

Our content is mostly static (prose and code examples), but there were a number of places within this static content where we needed to add interactivity (think things like the "Copy" button on code blocks). For these interactive parts, we ended up using regular DOM APIs, which wasn't very elegant, particularly when the rest of the site was written in React. We couldn't use JSX (React's HTML-like syntax), which limited the maintainability of more complex pieces of interactivity, and we occasionally faced the worst-case scenario of maintaining duplicate implementations - one using React and another using DOM APIs.'''
}
result=chain_result.invoke({"text": text})

print(result)

print(result.get_graph().draw_ascii())