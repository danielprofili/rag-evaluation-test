from rag import rag_ask

if __name__ == "__main__":
    print("\n--- PDF Chatbot Initialized ---")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        # 1. Wait for the user to type a question
        user_question = input("\nYour Question: ")

        # 2. Allow the user to break the loop and close the program
        if user_question.lower() in ['exit', 'quit']:
            print("Shutting down chatbot. Goodbye!")
            break

        # 3. Send the question through our RAG chain
        response = rag_ask(user_question)

        # 4. Clean up the output format
        # if isinstance(response.content, list):
        #     clean_answer = response.content[0]['text']
        # else:
        #     clean_answer = response.content

        # 5. Print the final answer to the console
        answer = response['answer'][0]['text']
        print(f"Answer: {answer}") 
