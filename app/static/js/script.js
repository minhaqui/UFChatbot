// Removemos todos os console.log

async function handleSubmit(event) {
    event.preventDefault();
    
    const mensagem = document.getElementById('mensagem').value;
    const respostaA = document.getElementById('resposta-a');
    const respostaB = document.getElementById('resposta-b');

    // Converte markdown para HTML
    respostaA.innerHTML += `<div class="user-message">Você: ${marked.parse(mensagem)}</div>`;
    respostaB.innerHTML += `<div class="user-message">Você: ${marked.parse(mensagem)}</div>`;

    const response = await fetch('/send_message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: mensagem })
    });

    const data = await response.json();

    respostaA.innerHTML += `<div class="model-message">Modelo A: ${marked.parse(data.resposta_a)}</div>`;
    respostaB.innerHTML += `<div class="model-message">Modelo B: ${marked.parse(data.resposta_b)}</div>`;

    document.getElementById('mensagem').value = '';

    // Exibe a seção de avaliação
    document.getElementById('avaliacao').style.display = 'block';
    checkFields(); // Alterado de checkProficiencia para checkFields
}

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('chat-form');
    if (form) {
        form.addEventListener('submit', handleSubmit);
    }
    
    // Adiciona eventos para verificar os campos em tempo real
    document.getElementById('nome').addEventListener('input', checkFields);
    document.getElementById('proficiencia').addEventListener('change', checkFields);
});

function checkFields() {
    const nome = document.getElementById('nome').value.trim();
    const prof = document.getElementById('proficiencia').value;
    const btnA = document.getElementById('btn-chat-a');
    const btnB = document.getElementById('btn-chat-b');
    const aviso = document.getElementById('aviso-proficiencia');
    
    if (nome === "" || prof === "") {
        btnA.disabled = true;
        btnB.disabled = true;
        btnA.classList.add('btn-disabled');
        btnB.classList.add('btn-disabled');
        aviso.textContent = "Por favor, preencha o nome e selecione seu nível de proficiência para votar.";
        aviso.style.display = 'block';
    } else {
        btnA.disabled = false;
        btnB.disabled = false;
        btnA.classList.remove('btn-disabled');
        btnB.classList.remove('btn-disabled');
        aviso.style.display = 'none';
    }
}

function avaliar(modelo) {
    const nome = document.getElementById('nome').value.trim();
    const prof = document.getElementById('proficiencia').value;
    
    // Verifica se os campos obrigatórios estão preenchidos
    if (!nome || !prof) {
        const feedback = document.getElementById('feedback');
        feedback.innerHTML = "<span class='erro'>Você precisa preencher o nome e selecionar seu nível de proficiência para avaliar.</span>";
        feedback.style.display = 'block';
        return;
    }
    
    const email = document.getElementById('email').value.trim();

    fetch('/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ winner: modelo, nome, email, proficiencia: prof })
    })
    .then(response => response.json())
    .then(data => {
        const feedback = document.getElementById('feedback');
        feedback.innerHTML = "<span class='sucesso'>Avaliação registrada com sucesso! Obrigado.</span>";
        feedback.style.display = 'block';
        setTimeout(() => resetChat(), 2000);
    })
    .catch(error => {
        const feedback = document.getElementById('feedback');
        feedback.innerHTML = "<span class='erro'>Ocorreu um erro ao registrar sua avaliação. Tente novamente.</span>";
        feedback.style.display = 'block';
    });
}

function resetChat() {
    fetch('/reset', { method: 'POST' })
    .then(response => response.json())
    .then(data => {
        document.getElementById('mensagem').value = '';
        document.getElementById('resposta-a').innerHTML = '';
        document.getElementById('resposta-b').innerHTML = '';
        document.getElementById('avaliacao').style.display = 'none';
        document.getElementById('feedback').style.display = 'none';
    })
    .catch(error => {});
}

// Alternar menu hamburger
document.querySelector('.menu-toggle').addEventListener('click', function() {
    document.querySelector('.navbar ul').classList.toggle('show');
});