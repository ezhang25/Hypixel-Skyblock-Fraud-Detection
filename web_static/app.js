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

function addMetadata(flag, root) {
  const details = [];
  if (flag.item_rarity) details.push(`Rarity: ${flag.item_rarity}`);
  if (flag.category) details.push(`Category: ${flag.category}`);
  if (flag.decoded_stars != null) details.push(`Stars: ${flag.decoded_stars}`);
  if (flag.decoded_recombobulated) details.push('Recombobulated');
  if (flag.decoded_reforge) details.push(`Reforge: ${flag.decoded_reforge}`);
  if (flag.decoded_dungeon_tier != null) details.push(`Dungeon: F${flag.decoded_dungeon_tier}`);
  if (flag.decoded_hot_potato_count) details.push(`HPB: ${flag.decoded_hot_potato_count}`);
  if (flag.decoded_fuming_potato_count) details.push(`Fuming: ${flag.decoded_fuming_potato_count}`);
  if (flag.decoded_enchant_summary) details.push(`Enchants: ${flag.decoded_enchant_summary}`);
  if (flag.decoded_gemstone_summary) details.push(`Gems: ${flag.decoded_gemstone_summary}`);
  if (flag.decoded_attribute_summary) details.push(`Attributes: ${flag.decoded_attribute_summary}`);
  if (flag.decoded_rune_summary) details.push(`Rune: ${flag.decoded_rune_summary}`);
  if (flag.decoded_skin) details.push(`Skin: ${flag.decoded_skin}`);
  if (flag.decoded_dye) details.push(`Dye: ${flag.decoded_dye}`);
  if (flag.decoded_pet_type) {
    const pet = [flag.decoded_pet_tier, flag.decoded_pet_type, flag.decoded_pet_level != null ? `Lvl ${flag.decoded_pet_level}` : ''].filter(Boolean).join(' ');
    details.push(`Pet: ${pet}`);
  }
  if (flag.decoded_pet_held_item) details.push(`Pet item: ${flag.decoded_pet_held_item}`);
  details.forEach(detail => {
    const tag = document.createElement('span');
    tag.textContent = detail;
    root.append(tag);
  });
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
  const quantity = Math.max(1, Number(flag.item_quantity || 1));
  const format = flag.is_bin ? 'Buy It Now' : `Auction · ${flag.bid_count || 0} bids`;
  node.querySelector('.format').textContent = quantity > 1 ? `${format} · Quantity ${number.format(quantity)}` : format;
  addMetadata(flag, node.querySelector('.item-meta'));
  node.querySelector('.price').textContent = `${coins(flag.final_price)} coins`;
  if (quantity > 1) node.querySelector('.unit-price').textContent = `${coins(flag.final_price / quantity)} each`;
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
