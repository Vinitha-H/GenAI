## Inspiration
**Creating an AI-Powered Language Coach  LiguaCoach**
My journey into language learning began as an immigrant in the German-speaking part of Switzerland. Faced with the challenge of mastering a new language, I found myself relying on translators and dictionaries, yearning for a more immersive and practical learning experience. The idea of an always-available Language Coach—a tool that could assist in mastering syntax, grammar, and vocabulary within real-life contexts—was born from this personal struggle. This project is fueled by the belief that technology has the power to revolutionize language learning, making it more accessible, personalized, and enjoyable for individuals worldwide.

## What it does
The app has two components
- **Translate & Correct** - that accepts an optional scenario and text input and translates it to another language specified by the user or it validates the grammar of the text entered by the user and additionally provides explanations and possible alternative way of conveying the same message. It also outputs an audio recording of the response to assist with the pronunciation and accent. There is also a possibility to upload and transcribe an audio file along with the translation.
- **Learn via Chat** - this provides mostly the same functionality as described above, except for the audio response. But this provides a more interactive way of learning the language.

## How I built it
In addition to using python, I used Vertex AI for the GenAI model, Streamlit for the UI, and hosted it on Cloud Run, using a Docker image. 

## Challenges I ran into
Although Streamlit provides a lot of functionality to design the UI experience, designing it appropriately was an interesting challenge. Creating a web-based language coach posed numerous challenges, spanning technical complexities to conceptual hurdles. One of the primary challenges involved setting the right prompts to get the right responses, and to some extent, coordinating widget interactions and updates. Additionally, ensuring the accuracy and relevance of language coaching content demanded rigorous testing and refinement. Balancing functionality with simplicity and usability was an ongoing endeavor, requiring continuous iteration to enhance the user experience.

## Accomplishments that I'm proud of
To have been able to come up with a GenAI idea and be able to prototype it in 2 weeks!

## What I learned
Delving into the nuances of Vertex-AI Gemini prompt engineering, and user experience (UX) design has provided valuable insights into the intersection of technology and language education. Moreover, crafting intuitive and engaging AI chat interfaces has underscored the importance of catering to individual learning needs in creating impactful educational tools.

## What's next for LinguaCoach - Translate & Correct
While the current iteration serves as a prototype with limited functionality, the potential for expansion is vast. The next enhancement would have been the ability to also upload and translate documents and thereby offering better support while trying to learn a new language.
Imagine a Language Buddy—a fully realized language coach with a speech-based interface enabling users to practice real-life scenarios such as job interviews or sales pitches in different languages. This tool could aid in perfecting diction, intonation, and various speaking forms, fostering connection and understanding across cultures. Whether as a standalone start-up venture or as an integrated feature in existing language learning platforms, the possibilities for this project are boundless.

![LinguaCoach](https://github.com/Vinitha-H/GenAI/assets/168436312/a869f4cd-d468-44fe-bc15-e377c7857b10)

