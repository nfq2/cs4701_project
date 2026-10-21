const $ = id => document.getElementById(id);
let state = null;
let busy = false;
const suitNames = {S: 'spades', H: 'hearts'};
const suits = {S: '♠', H: '♥'};
const signed = value => value > 0 ? `+${value}` : String(value);
function card(id, value, concealed = false) {
  const el = $(id);
  el.replaceChildren();
  el.className = 'card';
  if (!value) {
    el.classList.add(concealed ? 'back' : 'empty');
    const mark = document.createElement('span');
    mark.textContent = concealed ? '♠' : '?';
    el.append(mark);
    el.setAttribute('aria-label', concealed ? 'Opponent card hidden' : 'Public card not dealt');
    return;
  }
  if (value[0] === 'H') el.classList.add('red');
  el.setAttribute('aria-label', `${value[1]} of ${suitNames[value[0]]}`);
  for (const [className, text] of [['rank', value[1]], ['suit', suits[value[0]]]]) {
    const span = document.createElement('span'); span.className = className; span.textContent = text; el.append(span);
  }
}
function actionButton(label, action, style = '', next = false) {
  const button = document.createElement('button');
  button.textContent = label; button.className = style; button.disabled = busy;
  button.addEventListener('click', () => send(next ? '/api/next' : '/api/action', {action}));
  $('actions').append(button);
}
function render() {
  if (!state) return;
  const picker = $('opponent-select');
  if (!picker.options.length) {
    for (const opponent of state.opponents) {
      const option = document.createElement('option');
      option.value = opponent.id; option.textContent = opponent.name; picker.append(option);
    }
  }
  if (!busy) picker.value = state.opponent;
  picker.disabled = busy;
  $('opponent-button').disabled = busy;
  $('restart-button').disabled = busy;
  $('bot-name').textContent = state.opponent_name;
  $('bot-avatar').textContent = state.opponent_name[0];
  $('opponent-description').textContent = state.opponents.find(o => o.id === state.opponent).description;
  card('human-card', state.human_card);
  card('bot-card', state.opponent_card, true);
  card('public-card', state.public_card);
  $('hand-label').textContent = `HAND ${String(state.hand).padStart(2, '0')}`;
  $('round-label').textContent = state.over ? 'HAND COMPLETE' : `ROUND ${state.round} OF 2`;
  $('pot').replaceChildren(document.createTextNode(`${state.pot} `));
  const small = document.createElement('small'); small.textContent = 'chips'; $('pot').append(small);
  $('blind').textContent = state.human_blind;
  $('human-bet').textContent = `In for ${state.human_contribution} chips`;
  $('bot-bet').textContent = `In for ${state.opponent_contribution} chips`;
  $('score').textContent = signed(state.total_chips);
  $('completed').textContent = state.completed;
  $('actions').replaceChildren();
  if (state.over) {
    $('status').textContent = state.payoff > 0 ? `You win ${state.payoff} chips.` : state.payoff < 0 ? `You lose ${-state.payoff} chips.` : 'A split pot. Even this hand.';
    $('hint').textContent = state.showdown ? 'Showdown. Both private cards are revealed.' : `${state.history.at(-1).actor === 'You' ? 'You folded' : 'The agent folded'}. Private cards stay private.`;
    actionButton('Deal next hand  →', null, 'primary', true);
  } else {
    $('status').textContent = 'Your move.';
    $('hint').textContent = state.to_call ? `${state.to_call} chips to call. Raise, match the bet, or fold.` : 'No bet to match. Check to continue or raise the stakes.';
    const labels = {fold:'Fold', check:'Check', call:`Call ${state.to_call}`, raise:`Raise +${state.raise_by}`};
    for (const action of ['fold','check','call','raise']) {
      if (state.legal_actions.includes(action)) actionButton(labels[action], action, action === 'raise' ? 'primary' : action === 'fold' ? 'fold' : '');
    }
  }
  $('history').replaceChildren();
  let round = 0;
  for (const event of state.history) {
    if (event.round !== round) {
      round = event.round;
      const divider = document.createElement('li'); divider.className = 'event-round'; divider.textContent = `ROUND ${round}`; $('history').append(divider);
    }
    const item = document.createElement('li');
    const verbs = {call:'called', raise:'raised', fold:'folded', check:'checked'};
    item.textContent = `${event.actor} ${verbs[event.action]}.`; $('history').append(item);
  }
  if (!state.history.length) {
    const item = document.createElement('li'); item.className = 'muted'; item.textContent = 'The blinds are in. You act first.'; $('history').append(item);
  }
}
async function load() {
  const response = await fetch('/api/state');
  if (!response.ok) throw new Error('Could not load the game. Refresh to try again.');
  state = await response.json(); render();
}
async function send(url, body) {
  if (busy || !state) return;
  busy = true; $('error').hidden = true; render();
  try {
    const response = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({...body, revision:state.revision})});
    const data = await response.json();
    if (!response.ok) { await load(); throw new Error(data.error || 'Could not apply that action.'); }
    state = data;
  } catch (error) {
    $('error').textContent = error.message || 'Connection lost. Check the server and refresh.';
    $('error').hidden = false;
  } finally { busy = false; render(); }
}
$('restart-button').addEventListener('click', () => send('/api/restart', {}));
$('opponent-button').addEventListener('click', () => send('/api/opponent', {opponent: $('opponent-select').value}));
$('rules-button').addEventListener('click', () => {
  $('rules').hidden = !$('rules').hidden;
  $('rules-button').setAttribute('aria-expanded', String(!$('rules').hidden));
});
load().catch(error => {
  $('status').textContent = 'The table is unavailable.';
  $('hint').textContent = 'Check that the Python server is running, then refresh.';
  $('error').textContent = error.message; $('error').hidden = false;
});
