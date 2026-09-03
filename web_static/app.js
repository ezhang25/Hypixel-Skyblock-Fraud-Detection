const number = new Intl.NumberFormat();
const flagsRoot = document.querySelector('#flags');
const status = document.querySelector('#status');
const tierFilter = document.querySelector('#tier-filter');

function coins(value) {
  if (!value) return '—';
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(2)}B`;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return number.format(value);
}

async function request(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || 'Request failed');
  return payload;
}

function renderFlag(flag) {
  const node = document.querySelector('#flag-template').content.cloneNode(true);
  const tier = node.querySelector('.tier');
  tier.textContent = flag.tier;
  tier.classList.add(flag.tier.toLowerCase());
  node.querySelector('.score').textContent = `Score ${Number(flag.anomaly_score || 0).toFixed(3)}`;
  node.querySelector('.item-name').textContent = flag.decoded_clean_name || flag.item_name;
  node.querySelector('.format').textContent = flag.is_bin ? 'Buy It Now' : `Auction · ${flag.bid_count || 0} bids`;
  node.querySelector('.price').textContent = `${coins(flag.final_price)} coins`;
  node.querySelector('.seller').textContent = `Seller: ${flag.seller_uuid || '—'}`;
  node.querySelector('.buyer').textContent = `Buyer: ${flag.buyer_uuid || '—'}`;
  const reasons = node.querySelector('.reasons');
  const items = flag.reasons?.length ? flag.reasons : ['Model anomaly score exceeded the flag threshold.'];
  items.forEach(reason => { const item = document.createElement('li'); item.textContent = reason; reasons.append(item); });
  const notes = node.querySelector('.notes');
  node.querySelector('.confirm').addEventListener('click', () => label(flag.auction_id, 1, notes));
  node.querySelector('.false-positive').addEventListener('click', () => label(flag.auction_id, 0, notes));
  flagsRoot.append(node);
}

async function label(auctionId, labelValue, notes) {
  const buttons = [...notes.parentElement.querySelectorAll('button')];
  buttons.forEach(button => button.disabled = true);
  try {
    const result = await request(`/api/flags/${encodeURIComponent(auctionId)}/label`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({label: labelValue, notes: notes.value}),
    });
    status.textContent = result.message;
    await load();
  } catch (error) { status.textContent = error.message; buttons.forEach(button => button.disabled = false); }
}

async function load() {
  status.textContent = 'Loading…';
  try {
    const suffix = tierFilter.value ? `?tier=${tierFilter.value}` : '';
    const [stats, queue] = await Promise.all([request('/api/stats'), request(`/api/flags${suffix}`)]);
    document.querySelector('#all-time').textContent = number.format(stats.all_time_auctions_ingested);
    document.querySelector('#retained').textContent = number.format(stats.total_auctions);
    document.querySelector('#unreviewed').textContent = number.format(stats.flagged_unreviewed);
    document.querySelector('#confirmed').textContent = number.format(stats.confirmed_irl);
    flagsRoot.innerHTML = '';
    queue.flags.forEach(renderFlag);
    if (!queue.flags.length) flagsRoot.innerHTML = '<tr><td class="empty" colspan="6">No unreviewed alerts. New model flags will appear here automatically.</td></tr>';
    status.textContent = `Updated ${new Date().toLocaleTimeString()}`;
  } catch (error) { status.textContent = `Could not load dashboard data: ${error.message}`; }
}

document.querySelector('#refresh').addEventListener('click', load);
tierFilter.addEventListener('change', load);
load();
