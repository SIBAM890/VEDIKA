document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const micBtn = document.getElementById('mic-btn');
    const welcomeContainer = document.getElementById('welcome-container');

    // API URL for our backend
    const API_URL = "http://127.0.0.1:8000/api/chat";
    const welcomeMessage = "Jai Gurudev! Welcome to Sri Sri University! I'm here to guide you through everything about the admissions process, academic programs, and student life. Let's explore SSU together! 🎓";

    // --- State Management ---
    let wasInputFromSpeech = false;
    let conversationStarted = false;

    // --- Core Functions ---
    function addMessage(text, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);

        // Safely check if the 'marked' library is available for bot messages
        if (sender === 'bot' && typeof marked !== 'undefined') {
             // Use marked.parse() to convert Markdown to HTML
             const cleanHtml = typeof DOMPurify !== 'undefined' ? DOMPurify.sanitize(marked.parse(text)) : marked.parse(text);
             messageElement.innerHTML = cleanHtml;
        } else {
            // For user messages or if marked.js fails to load, use plain text
            messageElement.textContent = text;
        }
        
        chatMessages.appendChild(messageElement);
        // Scroll smoothly to the bottom of the chat window
        chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
    }

    function showTypingIndicator() {
        const typingIndicator = document.createElement('div');
        typingIndicator.id = 'typing-indicator-dynamic';
        typingIndicator.classList.add('message', 'bot-message', 'typing-indicator');
        typingIndicator.innerHTML = '<span></span><span></span><span></span>';
        chatMessages.appendChild(typingIndicator);
        chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
        return typingIndicator.id;
    }

    function removeTypingIndicator(indicatorId) {
        const indicator = document.getElementById(indicatorId);
        if (indicator) {
            chatMessages.removeChild(indicator);
        }
    }

    function startConversation() {
        if (!conversationStarted) {
            welcomeContainer.style.display = 'none';
            // Align subsequent messages to the top
            chatMessages.style.justifyContent = 'flex-start';
            conversationStarted = true;
        }
    }

    async function handleUserMessage(message) {
        if (!message || message.trim() === '') return;

        startConversation();
        addMessage(message, 'user');
        userInput.value = '';

        const typingIndicatorId = showTypingIndicator();
        await getBotResponse(message, typingIndicatorId);
    }
    
    async function getBotResponse(userText, typingIndicatorId) {
        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: userText }),
            });
            
            removeTypingIndicator(typingIndicatorId);

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            const data = await response.json();
            addMessage(data.reply, 'bot');

            if (wasInputFromSpeech) {
                // Clean the text of Markdown for clearer speech
                const cleanText = data.reply.replace(/[*_`#~]/g, '');
                speakText(cleanText);
                wasInputFromSpeech = false; 
            }

        } catch (error) {
            console.error('Error fetching bot response:', error);
            removeTypingIndicator(typingIndicatorId);
            const errorMsg = 'Sorry, something went wrong connecting. Please try again.';
            addMessage(errorMsg, 'bot');
            if (wasInputFromSpeech) {
                speakText(errorMsg);
                wasInputFromSpeech = false; 
            }
        }
    }

    // --- Text-to-Speech (TTS) ---
    function speakText(text) {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            const setVoice = () => {
                const voices = window.speechSynthesis.getVoices();
                 if (voices.length === 0) {
                     setTimeout(setVoice, 100); // Retry if voices aren't loaded
                     return;
                 }
                const desiredVoice = voices.find(voice => voice.name === "Google हिन्दी");
                if (desiredVoice) utterance.voice = desiredVoice;
                utterance.onerror = (event) => console.error("SpeechSynthesis Error:", event);
                window.speechSynthesis.speak(utterance);
            };

            const voices = speechSynthesis.getVoices();
            if (voices.length > 0) setVoice();
            else speechSynthesis.onvoiceschanged = setVoice;
        }
    }

    // --- Speech-to-Text (STT) ---
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.interimResults = false;

        micBtn.addEventListener('click', () => {
            window.speechSynthesis.cancel();
            try {
                 recognition.start();
            } catch(e) {
                console.error("Recognition already started?");
            }
        });

        recognition.onstart = () => {
            micBtn.classList.add('listening');
            userInput.placeholder = "Listening...";
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            wasInputFromSpeech = true;
            handleUserMessage(transcript);
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
             userInput.placeholder = "Mic error. Try again?";
        };
        recognition.onend = () => {
            micBtn.classList.remove('listening');
            userInput.placeholder = "Ask Vedika about SSU...";
        };
    } else {
        micBtn.style.display = 'none';
    }

    // --- Event Listeners ---
    sendBtn.addEventListener('click', (e) => {
        e.preventDefault(); 
        handleUserMessage(userInput.value.trim());
    });
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault(); 
            handleUserMessage(userInput.value.trim());
        }
    });

    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const message = chip.getAttribute('data-message');
            wasInputFromSpeech = false;
            handleUserMessage(message);
        });
    });

    // --- Initial Load ---

    // Define the function to trigger the welcome speech
    function triggerWelcomeSpeech() {
        speakText(welcomeMessage);
    }

    // Add a one-time event listener to speak on the user's first interaction
    // This is the only welcome speech trigger, ensuring it runs only once.
    window.addEventListener('click', triggerWelcomeSpeech, { once: true });
    window.addEventListener('touchstart', triggerWelcomeSpeech, { once: true });

});

