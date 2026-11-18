/* LARVAE — physics layer (Jolt).
   One solver for everything that touches: floor, glasses, stones, the larva.
   Units: cm, g, s. Jolt's Y is up; the bench's floor plane is (x, y) → Jolt (x, z).

   The larva:
   - tail and middle are soft: short segments on stiff springs, drawn after a
     skeleton that the skills move. Where something presses on them they give
     way a little and spring back;
   - the head is hard: a rigid ball, moved by the skeleton, that nothing enters.
   Glasses are rigid, they do not bounce. They are moved only by the solver:
   by gravity, the floor, and what the larva's body, head and legs do to them.

   The skills do not place glasses. They move the skeleton (and, with the legs,
   push on a glass with a limited force); the solver decides the rest. */
(function (root) {
  'use strict';

  const LAYER_STATIC = 0, LAYER_MOVING = 1, LAYER_LARVA = 2, LAYER_GHOST = 3, NUM_LAYERS = 4;
  const G = 981;                         // cm/s²
  const GLASS_DENSITY = 2.5;             // g/cm³
  const GLASS_FRICTION = 0.6;

  let J = null;                          // the Jolt module, once loaded

  async function load(opts = {}) {
    if (J) return J;
    const factory = opts.factory || root.JoltFactory;
    J = await factory();
    root.Physics.ready = true;
    return J;
  }

  const v3 = (x, y, z) => new J.Vec3(x, y, z);
  const r3 = (x, y, z) => new J.RVec3(x, y, z);
  // rotate a vector by a quaternion (x, y, z, w)
  const qrot = (qx, qy, qz, qw, x, y, z) => {
    const tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
    return [x + qw * tx + (qy * tz - qz * ty), y + qw * ty + (qz * tx - qx * tz), z + qw * tz + (qx * ty - qy * tx)];
  };
  // bench (x, y, z-up) → Jolt (x, z-up, y)
  const toJ = (x, y, z) => [x, z, y];

  function createWorld(opts = {}) {
    if (!J) throw new Error('physics: call Physics.load() first');
    const settings = new J.JoltSettings();
    settings.mMaxWorkerThreads = opts.threads !== undefined ? opts.threads : (root.Physics.threads || 0);
    const olp = new J.ObjectLayerPairFilterTable(NUM_LAYERS);
    olp.EnableCollision(LAYER_STATIC, LAYER_MOVING);
    olp.EnableCollision(LAYER_MOVING, LAYER_MOVING);
    olp.EnableCollision(LAYER_LARVA, LAYER_STATIC);
    olp.EnableCollision(LAYER_LARVA, LAYER_MOVING);      // a larva's own parts do not collide; ghosts collide with nothing
    const bpi = new J.BroadPhaseLayerInterfaceTable(NUM_LAYERS, 2);
    bpi.MapObjectToBroadPhaseLayer(LAYER_STATIC, new J.BroadPhaseLayer(0));
    bpi.MapObjectToBroadPhaseLayer(LAYER_MOVING, new J.BroadPhaseLayer(1));
    bpi.MapObjectToBroadPhaseLayer(LAYER_LARVA, new J.BroadPhaseLayer(1));
    bpi.MapObjectToBroadPhaseLayer(LAYER_GHOST, new J.BroadPhaseLayer(1));
    settings.mObjectLayerPairFilter = olp;
    settings.mBroadPhaseLayerInterface = bpi;
    settings.mObjectVsBroadPhaseLayerFilter = new J.ObjectVsBroadPhaseLayerFilterTable(bpi, 2, olp, NUM_LAYERS);
    const jolt = new J.JoltInterface(settings);
    J.destroy(settings);
    const system = jolt.GetPhysicsSystem();
    const bi = system.GetBodyInterface();
    system.SetGravity(v3(0, -G, 0));
    // small things: contact tolerances in cm
    const ps = system.GetPhysicsSettings();
    ps.mPenetrationSlop = 0.003;
    ps.mLinearCastThreshold = 0.5;
    const T0 = Object.assign({}, opts.tune || {}, (typeof globalThis !== 'undefined' && globalThis.__TUNE) || {});
    ps.mManifoldTolerance = T0.manifoldTol || 0.05;
    ps.mSpeculativeContactDistance = T0.specDist || 0.3;
    ps.mNumVelocitySteps = T0.velSteps || 30;
    ps.mNumPositionSteps = T0.posSteps || 10;
    ps.mBaumgarte = T0.baumgarte || 0.8;
    ps.mDeterministicSimulation = true;
    // Jolt keeps a pair's contacts from the last step while the two have moved less than these
    // (meant for metres: 1 mm and 2° — here they would be 0.3 mm, and a contact that deep could go
    // unseen). A hundredth of a millimetre and a quarter degree here.
    // and no reuse of last step's contacts at all, unless asked: each step finds them afresh
    if (!T0.pairCache) ps.mUseBodyPairContactCache = false;
    if (!T0.oldCache) {
      ps.mBodyPairCacheMaxDeltaPositionSq = 0.001 * 0.001;
      ps.mBodyPairCacheCosMaxDeltaRotationDiv2 = Math.cos(0.25 * Math.PI / 180 / 2);
      ps.mContactNormalCosMaxDeltaRotation = Math.cos(1 * Math.PI / 180);
      ps.mContactPointPreserveLambdaMaxDistSq = 0.001 * 0.001;
    }
    // a body may sleep once its points move slower than this (cm/s) for mTimeBeforeSleep: the default
    // (0.03, meant as m/s) would be 0.3 mm/s here, which a piece resting on others never gets below
    ps.mPointVelocitySleepThreshold = T0.sleepV || 0.3;
    system.SetPhysicsSettings(ps);

    const W = { jolt, system, bi, glasses: [], stones: [], larvae: [], t: 0, names: new Map(), glassOf: new Map() };
    W.nameOf = k => W.names.get(k) || ('body ' + (k & 0x7fffff));
    // reused per step, so stepping allocates nothing in the solver's memory
    const TMP = { r: new J.RVec3(), v: new J.Vec3(), q: J.Quat.prototype.sIdentity(), r2: new J.Vec3() };
    W.tune = Object.assign({ springHz: 90, pushHz: 40, tailHz: 90, bodyScale: 2.67, dose: 0.1, tailDose: 0.15, substeps: 8, segContact: 0.3, flatS: 0.6, roundS: 3, legTiltHz: 8, legTorque: 10000, skinSlip: 0.2, setae: 1, glassCast: 0, softHead: 0, headDose: 0.05, depenCheck: 1, floorDepen: 1, liftGlass: 1 }, opts.tune || {}, (typeof globalThis !== 'undefined' && globalThis.__TUNE) || {});
    W.skinTouch = new Map();     // other body key → contacts with the larva's soft body this step
    // rigid contacts this step (the head, glasses, stones, floor): pairs of keys, depth, normal
    W.contacts = [];
    const cl = new J.ContactListenerJS();
    const rec = (b1, b2, man) => {
      const a = J.wrapPointer(b1, J.Body).GetID().GetIndexAndSequenceNumber(), b = J.wrapPointer(b2, J.Body).GetID().GetIndexAndSequenceNumber();
      for (const lv of W.larvae) {
        if (lv.segKeys.has(a)) W.skinTouch.set(b, (W.skinTouch.get(b) || 0) + 1);
        if (lv.segKeys.has(b)) W.skinTouch.set(a, (W.skinTouch.get(a) || 0) + 1);
      }
      const m = J.wrapPointer(man, J.ContactManifold), n = m.mWorldSpaceNormal;
      const p = m.GetWorldSpaceContactPointOn1(0);
      W.contacts.push({ a, b, depth: m.mPenetrationDepth, n: { x: n.GetX(), y: n.GetZ(), z: n.GetY() }, at: { x: p.GetX(), y: p.GetZ(), z: p.GetY() } });
    };
    cl.OnContactValidate = () => J.ValidateResult_AcceptAllContactsForThisBodyPair;
    // a segment of the soft body is backed by the skeleton through its spring, so in
    // a contact it counts as heavier than its own small mass: it holds a glass up
    // and still gives a little
    const scale = (b1, b2, io, man) => {
      const a = J.wrapPointer(b1, J.Body).GetID().GetIndexAndSequenceNumber(), b = J.wrapPointer(b2, J.Body).GetID().GetIndexAndSequenceNumber();
      const st = J.wrapPointer(io, J.ContactSettings);
      for (const lv of W.larvae) {
        // the snout, small and thin, bears what rides on it as if far heavier still
        const sc = k => lv.snout && k === lv.snout.key ? (W.tune.snoutContact || W.tune.segContact) : k === lv.headKey && W.tune.headContact ? W.tune.headContact : W.tune.segContact;
        if (lv.segKeys.has(a) || (lv.headAnchor && a === lv.headKey) || (lv.snout && a === lv.snout.key)) { st.mInvMassScale1 = sc(a); st.mInvInertiaScale1 = sc(a); }
        if (lv.segKeys.has(b) || (lv.headAnchor && b === lv.headKey) || (lv.snout && b === lv.snout.key)) { st.mInvMassScale2 = sc(b); st.mInvInertiaScale2 = sc(b); }
        // the skin is moist and smooth: a loose piece of glass lying partly on it and partly on
        // the floor stays where the floor holds it, and the larva slides on under it. A piece
        // the legs hold is held, and rubs on the skin as before.
        const mine = k => lv.segKeys.has(k) || k === lv.headKey || (lv.snout && k === lv.snout.key);
        const g = W.glassOf.get(mine(a) ? b : mine(b) ? a : -1);
        if (g && !g.grip) {
          const onSnout = lv.snout && (a === lv.snout.key || b === lv.snout.key);
          let skin = onSnout ? (W.tune.snoutSlip ?? W.tune.skinSlip) : W.tune.skinSlip;
          // her bristles all point from head to tail: a piece shifts along her back toward the tail
          // easily, toward the head it is caught (the friction of the skin depends on which way the
          // piece moves over it)
          const onHead = a === lv.headKey || b === lv.headKey;   // (the head is hard smooth chitin, no bristles)
          if (!onSnout && !onHead && W.tune.setae && man) { try {
            const part = J.wrapPointer(mine(a) ? b1 : b2, J.Body), gb = g.body, pt = man.GetWorldSpaceContactPointOn1(0), px = pt.GetX(), py = pt.GetY(), pz = pt.GetZ();
            const hq = lv.head.GetPosition(), hx = hq.GetX(), hz = hq.GetZ(), from = part.GetPosition();
            let ux = hx - from.GetX(), uz = hz - from.GetZ(); const ul = Math.hypot(ux, uz);   // (Jolt hands back one shared vector: read it at once)
            if (ul > 1e-4) { ux /= ul; uz /= ul;
                            const pv = (bd) => { let q = bd.GetCenterOfMassPosition(); const rx = px - q.GetX(), ry = py - q.GetY(), rz = pz - q.GetZ();
                q = bd.GetAngularVelocity(); const wx = q.GetX(), wy = q.GetY(), wz = q.GetZ(); q = bd.GetLinearVelocity();
                return [q.GetX() + wy * rz - wz * ry, q.GetZ() + wx * ry - wy * rx]; };
              const vg = pv(gb), vp = pv(part), along = (vg[0] - vp[0]) * ux + (vg[1] - vp[1]) * uz;
              if (along > (W.tune.setaeV ?? 0.02)) skin = W.tune.setaeAgainst ?? 1.0; } } catch (e) { if (!W.setaeErr) { W.setaeErr = 1; console.warn('setae:', e.message); } } }
          st.mCombinedFriction = Math.sqrt(skin * GLASS_FRICTION); }
      }
    };
    cl.OnContactAdded = (b1, b2, man, io) => { scale(b1, b2, io, J.wrapPointer(man, J.ContactManifold)); rec(b1, b2, man); };
    cl.OnContactPersisted = (b1, b2, man, io) => { scale(b1, b2, io, J.wrapPointer(man, J.ContactManifold)); rec(b1, b2, man); };
    cl.OnContactRemoved = () => {};
    system.SetContactListener(cl);

    // the floor
    {
      // no bigger than the bench: a huge box against small pieces loses precision
      // (thick, so nothing falling or tipping fast can pass through it)
      const shape = new J.BoxShapeSettings(v3(20, 5, 15), 0).Create().Get();
      const cs = new J.BodyCreationSettings(shape, r3(16, -5, 10), J.Quat.prototype.sIdentity(), J.EMotionType_Static, LAYER_STATIC);
      cs.mFriction = 0.8; cs.mRestitution = 0;
      W.floor = bi.CreateBody(cs); bi.AddBody(W.floor.GetID(), J.EActivation_DontActivate);
      W.floorId = W.floor.GetID().GetIndexAndSequenceNumber(); W.names.set(W.floorId, 'the floor');
      J.destroy(cs);
    }

    /* A glass: a slab of the given outline (cm, in its own frame) and thickness. */
    // Jolt hands a face to the contact clipper as at most 32 corners: a face with more
    // loses some, and a corner it loses can sink into the floor unseen. So an outline
    // keeps at most MAX_CORNERS: the corners that matter least (the flattest bends) go first.
    const MAX_CORNERS = 30, BEAD = 0.004;
    const fewerCorners = (local, max = MAX_CORNERS) => {
      const q = local.slice();
      const area = i => { const a = q[(i + q.length - 1) % q.length], b = q[i], c = q[(i + 1) % q.length];
        return Math.abs((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)); };
      while (q.length > max) { let k = 0, best = Infinity; for (let i = 0; i < q.length; i++) { const A = area(i); if (A < best) { best = A; k = i; } } q.splice(k, 1); }
      return q;
    };
    // the outline drawn in by d all round (the narrower face of a bevelled piece)
    const insetPoly = (poly, d) => {
      let A = 0; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) A += poly[j].x * poly[i].y - poly[i].x * poly[j].y;
      const sg = A > 0 ? 1 : -1;     // counter-clockwise: the inside is to the left of each edge
      return poly.map((q, i) => {
        const a = poly[(i + poly.length - 1) % poly.length], b = poly[(i + 1) % poly.length];
        let n1x = -(q.y - a.y), n1y = q.x - a.x, n2x = -(b.y - q.y), n2y = b.x - q.x;
        const l1 = Math.hypot(n1x, n1y) || 1, l2 = Math.hypot(n2x, n2y) || 1; n1x *= sg / l1; n1y *= sg / l1; n2x *= sg / l2; n2y *= sg / l2;
        let nx = n1x + n2x, ny = n1y + n2y; const ln = Math.hypot(nx, ny) || 1; nx /= ln; ny /= ln;
        const c = Math.max(0.35, nx * n1x + ny * n1y);       // a sharp corner moves in further, up to about three times
        const di = typeof d === 'function' ? d(q) : d;        // d may differ round the outline: edges of different slope
        return { x: q.x + nx * di / c, y: q.y + ny * di / c };
      });
    };
    /* opts.bevel (cm): the edges slope by this much, so a piece cannot stand on an edge; or a
       function of the outline point giving it there, for edges sloped differently round the piece.
       opts.flip: the narrow face is the underside (the piece lies "upside down"); else the top. */
    W.addGlass = (local, thick, x, y, rot, z = 0, opts = {}) => {
      local = fewerCorners(local, opts.corners || MAX_CORNERS);
      const bevel = opts.bevel || 0, narrow = bevel ? insetPoly(local, bevel) : local;
      const top = bevel && !opts.flip ? narrow : local;
      local = bevel && opts.flip ? narrow : local;          // g.local is always the underside
      const hs = new J.ConvexHullShapeSettings();
      for (const [i, q] of local.entries()) { hs.mPoints.push_back(v3(q.x, 0, q.y)); hs.mPoints.push_back(v3(top[i].x, thick, top[i].y)); }
      // glass has sharp edges: Jolt's rounding of a hull's edges (its convex radius) is kept to a
      // hundredth of a millimetre, or a body could sit inside an acute bevelled edge unseen
      hs.mMaxConvexRadius = W.tune.glassRound ?? 0.001; hs.mHullTolerance = 0.0005;
      hs.mDensity = GLASS_DENSITY;
      // and a tiny bead at every corner: a corner resting on something is then always
      // a contact the solver sees, whatever face the clipper chose
      const cs0 = new J.StaticCompoundShapeSettings();
      cs0.AddShape(v3(0, 0, 0), J.Quat.prototype.sIdentity(), hs, 0);
      const cx = local.reduce((a, q) => a + q.x, 0) / local.length, cy = local.reduce((a, q) => a + q.y, 0) / local.length;
      if (!opts.noBeads) for (const [i, q0] of local.entries()) for (const [q, h] of [[q0, BEAD], [top[i], thick - BEAD]]) {
        const dx = cx - q.x, dy = cy - q.y, d = Math.hypot(dx, dy) || 1, ix = q.x + dx / d * BEAD, iy = q.y + dy / d * BEAD;
        const ss = new J.SphereShapeSettings(BEAD); ss.mDensity = GLASS_DENSITY;
        cs0.AddShape(v3(ix, h, iy), J.Quat.prototype.sIdentity(), ss, 0);
      }
      const res = cs0.Create(); if (res.HasError()) throw new Error('glass hull: ' + res.GetError().c_str());
      const shape = res.Get();
      const q = J.Quat.prototype.sRotation(v3(0, 1, 0), -rot);   // bench rot is counter-clockwise seen from above
      const cs = new J.BodyCreationSettings(shape, r3(x, z, y), q, J.EMotionType_Dynamic, LAYER_MOVING);
      cs.mFriction = GLASS_FRICTION; cs.mRestitution = 0; cs.mLinearDamping = 0.2; cs.mAngularDamping = 0.4;
      cs.mMotionQuality = W.tune.glassCast ? J.EMotionQuality_LinearCast : J.EMotionQuality_Discrete;
      cs.mUseManifoldReduction = !!opts.reduce;   // as a rule keep every contact point: it rocks on its edges
      cs.mAllowSleeping = !!opts.sleep;     // as a rule never put to sleep: a glass balanced on an edge must fall over, not freeze there
      const body = bi.CreateBody(cs); bi.AddBody(body.GetID(), J.EActivation_Activate);
      J.destroy(cs);
      const g = { body, id: body.GetID(), local, localTop: top, bevel, thick };
      g.key = g.id.GetIndexAndSequenceNumber(); W.names.set(g.key, 'a glass'); W.glassOf.set(g.key, g);
      W.glasses.push(g);
      return g;
    };
    // where a glass is: its outline's points at the bottom and top, in bench coordinates
    W.glassPose = g => {
      const p = g.body.GetPosition(), r = g.body.GetRotation();
      const px = p.GetX(), py = p.GetY(), pz = p.GetZ(), qx = r.GetX(), qy = r.GetY(), qz = r.GetZ(), qw = r.GetW();
      const at = (lx, ly, lz) => { const w = qrot(qx, qy, qz, qw, lx, ly, lz); return { x: w[0] + px, y: w[2] + pz, z: w[1] + py }; };
      return { bottom: g.local.map(q => at(q.x, 0, q.y)), top: (g.localTop || g.local).map(q => at(q.x, g.thick, q.y)) };
    };

    /* The larva. Its soft body (tail and middle) is a chain of short rounded
       segments, each held to its place on the skeleton by a stiff spring: the
       skills move the skeleton; the segments follow, and where something presses
       on them they give a little and spring back (a glass on the tail presses it
       down about half a millimetre). The head is hard: a ball the skeleton moves
       outright, that nothing enters. Parts of one larva do not collide with each
       other. */
    W.addLarva = (dims) => {
      const { TAIL_LEN, TAIL_D, MID_LEN, MID_D, HEAD_D } = dims;
      const L = TAIL_LEN + MID_LEN;
      // a sausage of the given length, its cross-section an ellipse h high and w wide
      const CAP_STEP = 15;
      const sausage = (len, h, w) => {
        const hs = new J.ConvexHullShapeSettings();
        // in the segment's frame x is up and down, z sideways: h is its height, w its width
        const ring = (y, k) => { for (let i = 0; i < 16; i++) { const a = i / 16 * 2 * Math.PI; hs.mPoints.push_back(v3(Math.cos(a) * h * k, y, Math.sin(a) * w * k)); } };
        ring(-len / 2, 1); ring(len / 2, 1);
        // the round ends, every CAP_STEP degrees up to the tip, so the cap stays close to a half-ball
        const r = Math.min(h, w);
        for (let c = CAP_STEP; c < 90; c += CAP_STEP) { const a = c * Math.PI / 180; ring(-len / 2 - r * Math.sin(a), Math.cos(a)); ring(len / 2 + r * Math.sin(a), Math.cos(a)); }
        hs.mPoints.push_back(v3(0, -len / 2 - r, 0)); hs.mPoints.push_back(v3(0, len / 2 + r, 0));
        hs.mMaxConvexRadius = 0.01; hs.mHullTolerance = 0.0005;
        return hs.Create().Get();
      };
      // the soft tail, pressed from above, gets thinner and longer: it stays round, its
      // thickness goes from TAIL_D (in the open) to TAIL_FLAT_D (under a glass), and it
      // keeps its length as it thins (the tail never grows longer)
      const FLAT = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1];   // small steps: each change of shape is a small one
      const FLAT_D = dims.TAIL_FLAT_D || MID_D;
      const tailH = f => TAIL_D / 2 + (FLAT_D / 2 - TAIL_D / 2) * f, tailW = tailH;
      // With W.tune.midSquash the middle is soft too: pressed from above it gives, lower and a
      // little wider (its cross-section an ellipse of the same area), up to that share of its height
      const SQ = W.tune.midSquash || 0;
      const midH = f => MID_D / 2 * (1 - SQ * f), midW = f => MID_D / 2 / (1 - SQ * f);
      const tailK = f => 1;                                        // how much longer it is: not at all
      // ...but the larva can draw its tail in (a crawl like a caterpillar's): down to 70 % of its
      // length, in small steps, each a shape of its own
      const LENS = []; for (let k = 0.70; k <= 1.0001; k += 0.025) LENS.push(+k.toFixed(3));
      // segments: three on the tail, two on the middle; from the tail's end forward
      const cuts = [0, TAIL_LEN / 3, 2 * TAIL_LEN / 3, TAIL_LEN, TAIL_LEN + MID_LEN / 2, L];
      // a larva of about 2 g (tail 1.31, middle 0.62, head 0.07)
      const massOf = (s0, s1) => (W.tune.bodyScale || 1) * (s1 <= TAIL_LEN ? 0.49 / TAIL_LEN : 0.23 / MID_LEN) * (s1 - s0);
      const segs = [];
      for (let i = 0; i + 1 < cuts.length; i++) {
        const s0 = cuts[i], s1 = cuts[i + 1], len = s1 - s0, isTail = s1 <= TAIL_LEN, r = isTail ? TAIL_D / 2 : MID_D / 2;
        // as long as the segment, its rounded ends overlapping the next one's
        const shapes = isTail ? FLAT.map(f => LENS.map(k => sausage(len * tailK(f) * k, tailH(f), tailW(f))))
          : SQ ? FLAT.map(f => [sausage(len, midH(f), midW(f))]) : [[new J.CapsuleShapeSettings(Math.max(0.01, len / 2), r).Create().Get()]];
        const shape = isTail ? shapes[0][LENS.length - 1] : shapes[0][0];
        const cs = new J.BodyCreationSettings(shape, r3(0, r + 2, 0), J.Quat.prototype.sIdentity(), J.EMotionType_Dynamic, LAYER_LARVA);
        cs.mFriction = 0.7; cs.mRestitution = 0; cs.mLinearDamping = 2; cs.mAngularDamping = 4;
        cs.mOverrideMassProperties = J.EOverrideMassProperties_CalculateInertia; cs.mMassPropertiesOverride.mMass = massOf(s0, s1);
        cs.mMotionQuality = J.EMotionQuality_LinearCast; cs.mAllowSleeping = false;
        const body = bi.CreateBody(cs); bi.AddBody(body.GetID(), J.EActivation_Activate); J.destroy(cs);
        // its anchor on the skeleton: moved outright, touches nothing
        const as = new J.SphereShapeSettings(0.01).Create().Get();
        const acs = new J.BodyCreationSettings(as, r3(0, r + 2, 0), J.Quat.prototype.sIdentity(), J.EMotionType_Kinematic, LAYER_GHOST);
        const anchor = bi.CreateBody(acs); bi.AddBody(anchor.GetID(), J.EActivation_Activate); J.destroy(acs);
        const c6 = new J.SixDOFConstraintSettings();
        c6.mSpace = J.EConstraintSpace_LocalToBodyCOM;
        for (const ax of [J.SixDOFConstraintSettings_EAxis_TranslationX, J.SixDOFConstraintSettings_EAxis_TranslationY, J.SixDOFConstraintSettings_EAxis_TranslationZ,
          J.SixDOFConstraintSettings_EAxis_RotationX, J.SixDOFConstraintSettings_EAxis_RotationY, J.SixDOFConstraintSettings_EAxis_RotationZ]) {
          c6.SetLimitedAxis(ax, 0, 0);
          // the segment's own frame: x is up and down (the anchor turns +Y onto the body's axis
          // about a level line), y along the body, z sideways. It bears weight stiffly; pushing
          // along or sideways, and twisting, its muscles give way sooner
          const firm = ax === J.SixDOFConstraintSettings_EAxis_TranslationX;
          const sp = c6.get_mLimitsSpringSettings(ax); sp.mFrequency = firm ? W.tune.springHz : (isTail ? W.tune.tailHz : W.tune.pushHz); sp.mDamping = 1;
        }
        const con = c6.Create(anchor, body); system.AddConstraint(con); J.destroy(c6);
        const seg = { body, anchor, s0, s1, cs0: s0, cs1: s1, r, len, curLen: len, isTail, shapes, flat: 0, lenAt: LENS.length - 1, key: body.GetID().GetIndexAndSequenceNumber(), placed: false };
        W.names.set(seg.key, s1 <= TAIL_LEN ? 'the tail' : 'the middle');
        segs.push(seg);
      }
      const hs = new J.SphereShapeSettings(HEAD_D / 2).Create().Get();
      // As a rule the head is moved outright by the skeleton. With W.tune.softHead it is a body of
      // its own (hard chitin, 0.08 g) held to the skeleton by a stiff spring like the segments, and
      // dosed like them: against something that will not give it presses only so hard.
      const soft = !!W.tune.softHead;
      const hcs = new J.BodyCreationSettings(hs, r3(0, 3, 0), J.Quat.prototype.sIdentity(), soft ? J.EMotionType_Dynamic : J.EMotionType_Kinematic, LAYER_LARVA);
      hcs.mFriction = 0.1; hcs.mRestitution = 0;    // smooth hard chitin
      if (soft) { hcs.mOverrideMassProperties = J.EOverrideMassProperties_CalculateInertia; hcs.mMassPropertiesOverride.mMass = W.tune.headMass || 0.08;
        hcs.mLinearDamping = 2; hcs.mAngularDamping = 4; hcs.mMotionQuality = J.EMotionQuality_LinearCast; hcs.mAllowSleeping = false; }
      const head = bi.CreateBody(hcs); bi.AddBody(head.GetID(), J.EActivation_Activate); J.destroy(hcs);
      let headAnchor = null;
      if (soft) {
        const as = new J.SphereShapeSettings(0.01).Create().Get();
        const acs = new J.BodyCreationSettings(as, r3(0, 3, 0), J.Quat.prototype.sIdentity(), J.EMotionType_Kinematic, LAYER_GHOST);
        headAnchor = bi.CreateBody(acs); bi.AddBody(headAnchor.GetID(), J.EActivation_Activate); J.destroy(acs);
        const c6 = new J.SixDOFConstraintSettings(); c6.mSpace = J.EConstraintSpace_LocalToBodyCOM;
        for (const ax of [J.SixDOFConstraintSettings_EAxis_TranslationX, J.SixDOFConstraintSettings_EAxis_TranslationY, J.SixDOFConstraintSettings_EAxis_TranslationZ]) {
          c6.SetLimitedAxis(ax, 0, 0); const sp = c6.get_mLimitsSpringSettings(ax); sp.mFrequency = W.tune.springHz; sp.mDamping = 1; }
        for (const ax of [J.SixDOFConstraintSettings_EAxis_RotationX, J.SixDOFConstraintSettings_EAxis_RotationY, J.SixDOFConstraintSettings_EAxis_RotationZ]) c6.MakeFreeAxis(ax);
        const con = c6.Create(headAnchor, head); system.AddConstraint(con); J.destroy(c6);
      }
      // With W.tune.snout the front of the head is a low cone, its tip on the floor: the jaws and
      // the face, which the larva drives in under whatever lies ahead. It is a body of its own,
      // held level and turned the way the head looks; it is pushed only as hard as the larva can
      // push (W.tune.snout, dyn): an edge it meets rides up its slope, or does not.
      let snout = null;
      if (W.tune.snout) {
        const SL = W.tune.snoutLen || 1.0, hs2 = new J.ConvexHullShapeSettings();
        // in its own frame: x forward, y up, z sideways; the back sits at the head's front
        // a true cone: round in section like the body, lying on the floor, so that next to the
        // floor its sides draw in and it never wedges flat into the narrow gap under an edge
        for (const [x, rr] of [[0, 0.22], [0.25 * SL, 0.18], [0.5 * SL, 0.13], [0.75 * SL, 0.08], [SL, 0.035]])
          for (let i = 0; i < 16; i++) { const a = i * Math.PI / 8; hs2.mPoints.push_back(v3(x, rr + rr * Math.sin(a), rr * Math.cos(a))); }
        hs2.mMaxConvexRadius = 0.001; hs2.mDensity = 0.1;
        const sshape = hs2.Create().Get();
        const scs = new J.BodyCreationSettings(sshape, r3(0, 5, 0), J.Quat.prototype.sIdentity(), J.EMotionType_Dynamic, LAYER_LARVA);
        scs.mFriction = 0.1; scs.mRestitution = 0; scs.mLinearDamping = 2; scs.mAngularDamping = 4;
        scs.mOverrideMassProperties = J.EOverrideMassProperties_CalculateInertia; scs.mMassPropertiesOverride.mMass = W.tune.snoutMass || 0.1;
        scs.mMotionQuality = J.EMotionQuality_LinearCast; scs.mAllowSleeping = false;
        // Jolt guards against passing through only along lines it has already found; a turn sweeps
        // the tip sideways where none was found. So the snout turns and moves no faster than the
        // larva does (a few cm/s, about a radian a second)
        scs.mMaxAngularVelocity = W.tune.snoutMaxW || 1.5; scs.mMaxLinearVelocity = W.tune.snoutMaxV || 5;
        const sb = bi.CreateBody(scs); bi.AddBody(sb.GetID(), J.EActivation_Activate); J.destroy(scs);
        const as3 = new J.SphereShapeSettings(0.01).Create().Get();
        const acs3 = new J.BodyCreationSettings(as3, r3(0, 5, 0), J.Quat.prototype.sIdentity(), J.EMotionType_Kinematic, LAYER_GHOST);
        const sa = bi.CreateBody(acs3); bi.AddBody(sa.GetID(), J.EActivation_Activate); J.destroy(acs3);
        snout = { body: sb, anchor: sa, key: sb.GetID().GetIndexAndSequenceNumber(), L: SL, placed: false, con: null };
      }
      // the height of the axis above the floor at s: the tail's is lower where it is flattened
      const radius = s => { for (const sg of segs) if (s <= (sg.isTail ? sg.cs1 : sg.s1) + 1e-9) return sg.isTail ? tailH(FLAT[sg.flat]) : midH(FLAT[sg.flat]); return MID_D / 2; };
      const larva = { segs, head, headAnchor, L, radius, tailH, tailW, midH, midW, soft: !!SQ, tailK, FLAT, LENS, lenScale: 1, headR: HEAD_D / 2, headKey: head.GetID().GetIndexAndSequenceNumber(), segKeys: new Set(segs.map(q => q.key)) };
      larva.keys = new Set([...larva.segKeys, larva.headKey]);
      if (snout) { larva.snout = snout; larva.keys.add(snout.key); W.names.set(snout.key, 'the snout'); }
      W.names.set(larva.headKey, 'the head');
      W.larvae.push(larva);
      return larva;
    };

    // a rotation taking Jolt's +Y (a segment's axis) onto the direction d, with the
    // segment's own x kept as near to straight up as the axis allows (its height
    // stays its height whichever way the body points)
    const quatYto = (dx, dy, dz) => {
      const n = Math.hypot(dx, dy, dz) || 1; dx /= n; dy /= n; dz /= n;
      let xx = -dy * dx, xy = 1 - dy * dy, xz = -dy * dz;      // up, less its part along the axis
      let xn = Math.hypot(xx, xy, xz);
      if (xn < 1e-6) { xx = 1; xy = 0; xz = 0; xn = 1; }
      xx /= xn; xy /= xn; xz /= xn;
      const zx = xy * dz - xz * dy, zy = xz * dx - xx * dz, zz = xx * dy - xy * dx;   // z = x × y
      // rotation matrix with columns x, y=d, z → quaternion
      const m00 = xx, m01 = dx, m02 = zx, m10 = xy, m11 = dy, m12 = zy, m20 = xz, m21 = dz, m22 = zz, tr = m00 + m11 + m22;
      let qx, qy, qz, qw;
      if (tr > 0) { const S = Math.sqrt(tr + 1) * 2; qw = S / 4; qx = (m21 - m12) / S; qy = (m02 - m20) / S; qz = (m10 - m01) / S; }
      else if (m00 > m11 && m00 > m22) { const S = Math.sqrt(1 + m00 - m11 - m22) * 2; qw = (m21 - m12) / S; qx = S / 4; qy = (m01 + m10) / S; qz = (m02 + m20) / S; }
      else if (m11 > m22) { const S = Math.sqrt(1 + m11 - m00 - m22) * 2; qw = (m02 - m20) / S; qx = (m01 + m10) / S; qy = S / 4; qz = (m12 + m21) / S; }
      else { const S = Math.sqrt(1 + m22 - m00 - m11) * 2; qw = (m10 - m01) / S; qx = (m02 + m20) / S; qy = (m12 + m21) / S; qz = S / 4; }
      return [qx, qy, qz, qw];
    };

    /* Move the skeleton. spine(s) gives, for arc length s from the tail's end,
       the axis point (bench coordinates, z up) and the direction along the body.
       head: the head's centre. Placed outright the first time, then moved over dt. */
    W.setPose = (larva, spine, headAt, dt) => {
      // a thinned tail is longer: its segments take more of the spine, counted from the
      // tail–middle joint backwards (the spine runs on straight past the tail's end)
      { let cur = null; for (let i = larva.segs.length - 1; i >= 0; i--) { const sg = larva.segs[i]; if (!sg.isTail) continue;
          if (cur === null) cur = sg.s1; sg.curLen = sg.len * larva.tailK(larva.FLAT[sg.flat]) * larva.LENS[sg.lenAt]; sg.cs1 = cur; sg.cs0 = cur - sg.curLen; cur = sg.cs0; } }
      for (const sg of larva.segs) {
        // the segment lies along the spine's direction at its middle, its axis at the segment's own height
        const a = spine(sg.cs0), b = spine(sg.cs1), m = spine((sg.cs0 + sg.cs1) / 2);
        const pm = toJ(m.x, m.y, m.z), d = toJ(b.x - a.x, b.y - a.y, (b.z - a.z) - ((b.zr || 0) - (a.zr || 0)));
        const q = quatYto(d[0], d[1], d[2]);
        TMP.r.Set(pm[0], pm[1], pm[2]); sg.want = pm;
        const Q = new J.Quat(q[0], q[1], q[2], q[3]);
        if (sg.placed) {
          // she is strong but doses her force: the skeleton never runs further ahead of the
          // body than 'dose', so against something that will not give she presses only so hard
          const bp = sg.body.GetPosition(), ex = pm[0] - bp.GetX(), ey = pm[1] - bp.GetY(), ez = pm[2] - bp.GetZ(), e = Math.hypot(ex, ey, ez);
          // (a tail segment pressed flat as far as it goes by something lying on it is led on only
          // gently: it is not forced on into a gap that narrows, it is drawn through it)
          const squashed = sg.isTail && W.tune.pressedDose && sg.press >= 0.999 && W.contacts.some(c => (c.a === sg.key || c.b === sg.key) && W.glassOf.has(c.a === sg.key ? c.b : c.a) && c.depth > -0.01);
          const dose = squashed ? W.tune.pressedDose : sg.isTail ? W.tune.tailDose : W.tune.dose;
          let tx = bp.GetX() + ex * Math.min(1, dose / (e || 1)), ty = bp.GetY() + ey * Math.min(1, dose / (e || 1)), tz = bp.GetZ() + ez * Math.min(1, dose / (e || 1));
          // With W.tune.yieldSide the middle, light and soft, gives way to a glass leaning on its
          // side: it does not hold its place against it, it lets itself be pushed aside (in the
          // floor's plane only: it still bears weight)
          if (W.tune.yieldSide && !sg.isTail) for (const c of W.contacts) {
            const o = c.a === sg.key ? c.b : c.b === sg.key ? c.a : null;
            if (o === null || !W.glassOf.has(o)) continue;
            // (yieldSide 3: only to a glass that bears another glass — a stack's bottom, too heavy to be
            // pushed aside, so the soft body is the one that gives)
            if (W.tune.yieldSide === 3 && !W.contacts.some(e => { const q = e.a === o ? e.b : e.b === o ? e.a : null; return q !== null && W.glassOf.has(q) && Math.abs(e.n.z) > 0.5 && e.at.z > c.at.z; })) continue;
            // the way the glass pushes it, level: from the touch to the segment's axis (Jolt x, z)
            let px = bp.GetX() - c.at.x, pz = bp.GetZ() - c.at.y;
            // (yieldSide 2: only across her body — along it, her stroke is her own and still goes on)
            if (W.tune.yieldSide >= 2) { const al = Math.hypot(d[0], d[2]); if (al > 1e-6) { const ax = d[0] / al, az = d[2] / al, k = px * ax + pz * az; px -= k * ax; pz -= k * az; } }
            const pl = Math.hypot(px, pz); if (pl < 0.5 * sg.r) continue; px /= pl; pz /= pl;
            const pull = (tx - bp.GetX()) * px + (tz - bp.GetZ()) * pz;
            if (pull < 0) { const yk = W.tune.yieldK ?? 1; tx -= yk * pull * px; tz -= yk * pull * pz; }
          }
          TMP.r.Set(tx, ty, tz);
          bi.MoveKinematic(sg.anchor.GetID(), TMP.r, Q, dt);
        }
        else {
          bi.SetPositionAndRotation(sg.anchor.GetID(), TMP.r, Q, J.EActivation_Activate);
          bi.SetPositionAndRotation(sg.body.GetID(), TMP.r, Q, J.EActivation_Activate);
          sg.placed = true;
        }
        J.destroy(Q);
      }
      const hp = toJ(headAt.x, headAt.y, headAt.z);
      TMP.r.Set(hp[0], hp[1], hp[2]);
      if (larva.headAnchor) {
        if (larva.headPlaced) {
          const bp = larva.head.GetPosition(), ex = hp[0] - bp.GetX(), ey = hp[1] - bp.GetY(), ez = hp[2] - bp.GetZ(), e = Math.hypot(ex, ey, ez);
          if (e > W.tune.headDose) { const k = W.tune.headDose / e; TMP.r.Set(bp.GetX() + ex * k, bp.GetY() + ey * k, bp.GetZ() + ez * k); }
          bi.MoveKinematic(larva.headAnchor.GetID(), TMP.r, TMP.q, dt);
        } else { bi.SetPosition(larva.headAnchor.GetID(), TMP.r, J.EActivation_Activate); bi.SetPosition(larva.head.GetID(), TMP.r, J.EActivation_Activate); larva.headPlaced = true; }
      } else if (larva.headPlaced) bi.MoveKinematic(larva.head.GetID(), TMP.r, TMP.q, dt);
      else { bi.SetPosition(larva.head.GetID(), TMP.r, J.EActivation_Activate); larva.headPlaced = true; }
      // the snout: just ahead of the head's middle, on the floor, level, turned the way the head looks
      const sn = larva.snout;
      if (sn && headAt.dir) {
        const yaw = Math.atan2(headAt.dir.y, headAt.dir.x), back = larva.headR * 0.2;
        const sx = headAt.x + headAt.dir.x * back, sy = headAt.y + headAt.dir.y * back, sz = W.tune.snoutLift ?? 0.03;   // a hair over the floor
        const Q = J.Quat.prototype.sRotation(v3(0, 1, 0), -yaw);
        TMP.r.Set(sx, sz, sy); sn.want = { x: sx, y: sy };
        if (!sn.placed) {
          bi.SetPositionAndRotation(sn.anchor.GetID(), TMP.r, Q, J.EActivation_Activate);
          bi.SetPositionAndRotation(sn.body.GetID(), TMP.r, Q, J.EActivation_Activate);
          const c6 = new J.SixDOFConstraintSettings(); c6.mSpace = J.EConstraintSpace_LocalToBodyCOM;
          const TR = [J.SixDOFConstraintSettings_EAxis_TranslationX, J.SixDOFConstraintSettings_EAxis_TranslationY, J.SixDOFConstraintSettings_EAxis_TranslationZ];
          // forward and sideways it is driven by capped motors (below); up and down it is held to
          // the floor by a stiff spring, as the head bears on it
          c6.MakeFreeAxis(TR[0]); c6.MakeFreeAxis(TR[2]);
          if (W.tune.snoutRide) c6.MakeFreeAxis(TR[1]);
          else { c6.SetLimitedAxis(TR[1], 0, 0); const sp = c6.get_mLimitsSpringSettings(TR[1]); sp.mFrequency = W.tune.springHz; sp.mDamping = 1; }
          // it is kept level by stiff springs; it is turned with the head by a motor no stronger than
          // the larva (so swinging round against a piece beside it, it is held back by it)
          for (const ax of [J.SixDOFConstraintSettings_EAxis_RotationX, J.SixDOFConstraintSettings_EAxis_RotationZ]) {
            c6.SetLimitedAxis(ax, 0, 0); const sp = c6.get_mLimitsSpringSettings(ax); sp.mFrequency = W.tune.springHz; sp.mDamping = 1; }
          c6.MakeFreeAxis(J.SixDOFConstraintSettings_EAxis_RotationY);
          // the anchor's point and the snout's origin: same place in the world
          // (read each at once: Jolt hands back one shared vector for both)
          const c0 = sn.body.GetCenterOfMassPosition(), cx = c0.GetX(), cy = c0.GetY(), cz = c0.GetZ();
          const o0 = sn.body.GetPosition(), ox = o0.GetX(), oy = o0.GetY(), oz = o0.GetZ();
          c6.mPosition1 = v3(0, 0, 0);
          const rel = qrot(-Q.GetX(), -Q.GetY(), -Q.GetZ(), Q.GetW(), ox - cx, oy - cy, oz - cz);
          c6.mPosition2 = v3(rel[0], rel[1], rel[2]);
          sn.con = c6.Create(sn.anchor, sn.body); system.AddConstraint(sn.con); J.destroy(c6);
          // pushed and pulled along by motors no stronger than the larva
          const sc = J.castObject(sn.con, J.SixDOFConstraint);
          // (with W.tune.snoutRide it is held down too only as hard as the head bears on it, so it can
          // ride up over a piece lying under it rather than be pinned between two)
          for (const ax of W.tune.snoutRide ? TR : [TR[0], TR[2]]) { const ms = sc.GetMotorSettings(ax); ms.mSpringSettings.mFrequency = W.tune.snoutHz || 60; ms.mSpringSettings.mDamping = 1;
            ms.mMinForceLimit = -W.tune.snout; ms.mMaxForceLimit = W.tune.snout; sc.SetMotorState(ax, J.EMotorState_Position); }
          sc.SetTargetPositionCS(v3(0, 0, 0));
          { const ms = sc.GetMotorSettings(J.SixDOFConstraintSettings_EAxis_RotationY); ms.mSpringSettings.mFrequency = 20; ms.mSpringSettings.mDamping = 1;
            ms.mMinTorqueLimit = -W.tune.snout * (W.tune.snoutTorque || 0.5); ms.mMaxTorqueLimit = W.tune.snout * (W.tune.snoutTorque || 0.5); sc.SetMotorState(J.SixDOFConstraintSettings_EAxis_RotationY, J.EMotorState_Position); }
          sc.SetTargetOrientationCS(J.Quat.prototype.sIdentity());
          sn.placed = true;
        } else {
          // dosed like the rest: the anchor never runs further ahead of the snout than headDose
          const bp = sn.body.GetPosition(), ex = TMP.r.GetX() - bp.GetX(), ey = TMP.r.GetY() - bp.GetY(), ez = TMP.r.GetZ() - bp.GetZ(), e = Math.hypot(ex, ey, ez);
          if (e > W.tune.headDose * 3) { const k = W.tune.headDose * 3 / e; TMP.r.Set(bp.GetX() + ex * k, bp.GetY() + ey * k, bp.GetZ() + ez * k); }
          bi.MoveKinematic(sn.anchor.GetID(), TMP.r, Q, dt);
        }
        J.destroy(Q);
      }
    };

    // the soft body's surface as points (bench coordinates): rings round each segment
    // the soft body's surface as points (bench coordinates): rings round each segment and
    // its rounded ends, at the segment's true height and width (a pressed tail is wider)
    W.skinPoints = larva => {
      const out = [];
      for (const sg of larva.segs) {
        const p = sg.body.GetPosition(), r = sg.body.GetRotation(), qx = r.GetX(), qy = r.GetY(), qz = r.GetZ(), qw = r.GetW();
        const h = sg.isTail ? larva.tailH(larva.FLAT[sg.flat]) : larva.midH(larva.FLAT[sg.flat]), w = sg.isTail ? larva.tailW(larva.FLAT[sg.flat]) : larva.midW(larva.FLAT[sg.flat]), cap = Math.min(h, w);
        const rings = [];
        const ln = sg.curLen;
        for (let i = 0; i <= 4; i++) rings.push({ y: -ln / 2 + ln * i / 4, k: 1 });
        for (const c of [35, 65, 90]) { const a = c * Math.PI / 180; rings.push({ y: -ln / 2 - cap * Math.sin(a), k: Math.cos(a) }, { y: ln / 2 + cap * Math.sin(a), k: Math.cos(a) }); }
        for (const rg of rings) {
          const n = rg.k < 0.05 ? 1 : 12;
          for (let k = 0; k < n; k++) {
            const t = k / n * 2 * Math.PI, q = qrot(qx, qy, qz, qw, Math.cos(t) * h * rg.k, rg.y, Math.sin(t) * w * rg.k);
            const sAt = sg.cs0 + (sg.cs1 - sg.cs0) * Math.max(0, Math.min(1, (rg.y + ln / 2) / ln));
            out.push({ x: q[0] + p.GetX(), y: q[2] + p.GetZ(), z: q[1] + p.GetY(), s: sAt, seg: sg.key });
          }
        }
      }
      return out;
    };
    // how far each segment has been pushed off its place on the skeleton (cm)
    W.give = larva => larva.segs.map(sg => { const a = sg.anchor.GetPosition(), ax = a.GetX(), ay = a.GetY(), az = a.GetZ(), b = sg.body.GetPosition(); return Math.hypot(ax - b.GetX(), ay - b.GetY(), az - b.GetZ()); });
    W.headPoint = larva => { const p = larva.head.GetPosition(); return { x: p.GetX(), y: p.GetZ(), z: p.GetY() }; };

    W.pushGlass = (g, fx, fy, fz, at) => {   // a force (g·cm/s²) on a glass at a point (bench coordinates)
      const f = toJ(fx, fy, fz), p = toJ(at.x, at.y, at.z);
      TMP.v.Set(f[0], f[1], f[2]); TMP.r.Set(p[0], p[1], p[2]);
      bi.AddForce(g.id, TMP.v, TMP.r, J.EActivation_Activate);
    };
    // the legs drive a glass that lies on the body: a velocity, horizontal, and a turn about the vertical
    W.driveGlass = (g, vx, vy, w) => {
      const lv = bi.GetLinearVelocity(g.id), av = bi.GetAngularVelocity(g.id);
      TMP.v.Set(vx, lv.GetY(), vy); bi.SetLinearVelocity(g.id, TMP.v);
      TMP.v.Set(av.GetX(), -w, av.GetZ()); bi.SetAngularVelocity(g.id, TMP.v);
    };
    // the legs move a glass that lies on the body: with force, against the friction
    // of what it rests on, toward a horizontal velocity and a turn about the vertical;
    // at most maxF of force (and the torque that force gives at 1 cm)
    W.legsOnGlass = (g, vx, vy, w, maxF, dt) => {
      const v = g.body.GetLinearVelocity(), av = g.body.GetAngularVelocity(), mp = g.body.GetMotionProperties();
      const m = 1 / mp.GetInverseMass(), I = m * 1.2 * 1.2, mu = 0.7;
      const sp = Math.hypot(vx, vy);
      let fx = (vx - v.GetX()) * m / dt + (sp > 1e-6 ? vx / sp * mu * m * G : 0);
      let fz = (vy - v.GetZ()) * m / dt + (sp > 1e-6 ? vy / sp * mu * m * G : 0);
      const f = Math.hypot(fx, fz); if (f > maxF) { fx *= maxF / f; fz *= maxF / f; }
      let ty = (-w - av.GetY()) * I / dt + (Math.abs(w) > 1e-6 ? -Math.sign(w) * mu * m * G * 0.8 : 0);
      ty = Math.max(-maxF, Math.min(maxF, ty));
      TMP.v.Set(fx, 0, fz); TMP.r2.Set(0, ty, 0);
      bi.AddForceAndTorque(g.id, TMP.v, TMP.r2, J.EActivation_Activate);
    };
    /* The legs grip a glass that lies on the body: they hold it from sliding off
       and move it across the body. The glass is held round the vertical and in
       the floor's plane, through a stiff spring (the legs give a little); up and
       down, and its tilt, are left to what it rests on. */
    W.grip = (g, at) => {
      if (g.grip) return g.grip;
      // the legs hold the glass where they are (at: bench x, y), not at its middle: legs are only
      // so long, so a glass whose weight lies far from them pulls on them harder
      const c = g.body.GetCenterOfMassPosition();
      const p = new J.RVec3(at ? at.x : c.GetX(), c.GetY(), at ? at.y : c.GetZ());
      const as = new J.SphereShapeSettings(0.01).Create().Get();
      TMP.r.Set(p.GetX(), p.GetY(), p.GetZ());
      const yaw = W.glassAngle(g);
      const q = J.Quat.prototype.sRotation(v3(0, 1, 0), -yaw);
      const acs = new J.BodyCreationSettings(as, TMP.r, q, J.EMotionType_Kinematic, LAYER_GHOST);
      const hand = bi.CreateBody(acs); bi.AddBody(hand.GetID(), J.EActivation_Activate); J.destroy(acs);
      const c6 = new J.SixDOFConstraintSettings();
      c6.mSpace = J.EConstraintSpace_WorldSpace;
      c6.mPosition1 = r3(p.GetX(), p.GetY(), p.GetZ()); c6.mPosition2 = r3(p.GetX(), p.GetY(), p.GetZ());
      for (const ax of [J.SixDOFConstraintSettings_EAxis_TranslationX, J.SixDOFConstraintSettings_EAxis_TranslationZ, J.SixDOFConstraintSettings_EAxis_RotationY]) {
        c6.SetLimitedAxis(ax, 0, 0); const sp = c6.get_mLimitsSpringSettings(ax); sp.mFrequency = 30; sp.mDamping = 1;
      }
      for (const ax of [J.SixDOFConstraintSettings_EAxis_TranslationY, J.SixDOFConstraintSettings_EAxis_RotationX, J.SixDOFConstraintSettings_EAxis_RotationZ]) c6.MakeFreeAxis(ax);
      const con = c6.Create(hand, g.body); system.AddConstraint(con); J.destroy(c6);
      // the legs also hold the glass from tipping, but only as hard as legs can: a
      // glass that needs more than that to stay level tips however they hold it
      { const sc = J.castObject(con, J.SixDOFConstraint);
        for (const ax of [J.SixDOFConstraintSettings_EAxis_RotationX, J.SixDOFConstraintSettings_EAxis_RotationZ]) {
          const ms = sc.GetMotorSettings(ax); ms.mSpringSettings.mFrequency = W.tune.legTiltHz; ms.mSpringSettings.mDamping = 1;
          ms.mMinTorqueLimit = -W.tune.legTorque; ms.mMaxTorqueLimit = W.tune.legTorque;
          sc.SetMotorState(ax, J.EMotorState_Position);
        }
        sc.SetTargetOrientationCS(J.Quat.prototype.sIdentity()); }
      const r0 = hand.GetRotation();
      const local = W.worldToGlass(g, { x: p.GetX(), y: p.GetZ(), z: p.GetY() });
      g.grip = { hand, con, sc: J.castObject(con, J.SixDOFConstraint), q0: [r0.GetX(), r0.GetY(), r0.GetZ(), r0.GetW()], local, x: p.GetX(), y: p.GetZ(), z: p.GetY(), yaw };
      J.destroy(p);
      return g.grip;
    };
    // where on the glass the legs hold it, now (bench coordinates)
    W.gripPoint = g => { const L = g.grip.local; return W.glassToWorld(g, L[0], L[1], L[2]);
    };
    // move the gripping legs: the glass's centre to (x, y) and its turn to yaw, over dt
    W.moveGrip = (g, x, y, yaw, dt) => {
      const G = g.grip; G.x = x; G.y = y; G.yaw = yaw;
      TMP.r.Set(x, G.z, y);
      const q = J.Quat.prototype.sRotation(v3(0, 1, 0), -yaw);
      bi.MoveKinematic(G.hand.GetID(), TMP.r, q, dt);
    };
    /* The legs tip the glass they hold a little: by ang (radians) about the horizontal
       axis (ax, ay) in bench coordinates, as far as their strength allows. 0 holds it level. */
    W.gripTilt = (g, ax, ay, ang) => {
      const G = g.grip; if (!G) return;
      const n = Math.hypot(ax, ay) || 1, h = G.hand.GetRotation();
      // the axis in the constraint's own frame: q0 · conj(hand now) · axis
      const qm = (a, b) => [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
      const R = qm(G.q0, [-h.GetX(), -h.GetY(), -h.GetZ(), h.GetW()]);
      const v = qrot(R[0], R[1], R[2], R[3], ax / n, 0, ay / n), sh = Math.sin(ang / 2);
      const q = new J.Quat(v[0] * sh, v[1] * sh, v[2] * sh, Math.cos(ang / 2));
      G.sc.SetTargetOrientationCS(q); J.destroy(q);
    };
    // how hard the legs work to keep the glass from tipping: their torque this step, as a share of the most they have
    W.gripStrain = (g, dt) => {
      const G = g.grip; if (!G) return 0;
      const l = G.sc.GetTotalLambdaMotorRotation();
      return Math.hypot(l.GetX(), l.GetZ()) / (dt / W.tune.substeps) / W.tune.legTorque;
    };
    // a hand lowering a glass: how much of its weight it carries itself (0 held, 1 let go)
    W.setWeight = (g, f) => bi.SetGravityFactor(g.id, f);
    /* Jaws: they take a glass at one point (fixed in the glass) and hold that point
       where they are moved to, through a stiff spring; the glass is free to turn
       about it. */
    /* The jaws take a glass at a point and carry it where they are moved. Given opts.maxForce
       (dyn), they pull only that hard (muscles, not a vice): a piece heavier than that at its
       edge does not come up, and jw.lag() tells how far the held point hangs below the jaws. */
    W.jaws = (g, at, opts = {}) => {
      const as = new J.SphereShapeSettings(0.01).Create().Get();
      TMP.r.Set(at.x, at.z, at.y);
      const acs = new J.BodyCreationSettings(as, TMP.r, TMP.q, J.EMotionType_Kinematic, LAYER_GHOST);
      const hand = bi.CreateBody(acs); bi.AddBody(hand.GetID(), J.EActivation_Activate); J.destroy(acs);
      const c6 = new J.SixDOFConstraintSettings();
      c6.mSpace = J.EConstraintSpace_WorldSpace;
      c6.mPosition1 = r3(at.x, at.z, at.y); c6.mPosition2 = r3(at.x, at.z, at.y);
      const TR = [J.SixDOFConstraintSettings_EAxis_TranslationX, J.SixDOFConstraintSettings_EAxis_TranslationY, J.SixDOFConstraintSettings_EAxis_TranslationZ];
      if (!opts.maxForce) {
        // they hold that point firmly (through a stiff spring); the glass is free to turn about it
        for (const ax of TR) { c6.SetLimitedAxis(ax, 0, 0); const sp = c6.get_mLimitsSpringSettings(ax); sp.mFrequency = 20; sp.mDamping = 1; }
      } else for (const ax of TR) c6.MakeFreeAxis(ax);
      for (const ax of [J.SixDOFConstraintSettings_EAxis_RotationX, J.SixDOFConstraintSettings_EAxis_RotationY, J.SixDOFConstraintSettings_EAxis_RotationZ]) c6.MakeFreeAxis(ax);
      const con = c6.Create(hand, g.body); system.AddConstraint(con); J.destroy(c6);
      if (opts.maxForce) {
        // as strong as the larva is and no stronger: a position motor on each axis, its force capped
        const sc = J.castObject(con, J.SixDOFConstraint);
        // (opts.liftOnly: they only lift — the edge is free to slide along the jaws as the snout goes in)
        for (const ax of opts.liftOnly ? [TR[1]] : TR) { const ms = sc.GetMotorSettings(ax); ms.mSpringSettings.mFrequency = 20; ms.mSpringSettings.mDamping = 1;
          ms.mMinForceLimit = opts.liftOnly ? 0 : -opts.maxForce; ms.mMaxForceLimit = opts.maxForce; sc.SetMotorState(ax, J.EMotorState_Position); }
        sc.SetTargetPositionCS(v3(0, 0, 0));
      }
      const held = W.worldToGlass(g, { x: at.x, y: at.y, z: at.z });
      const jw = { hand, con, x: at.x, y: at.y, z: at.z };
      // how far the held point of the glass hangs below where the jaws are (cm)
      jw.lag = () => { const q = W.glassToWorld(g, held[0], held[1], held[2]); return jw.z - q.z; };
      jw.move = (x, y, z, dt) => { jw.x = x; jw.y = y; jw.z = z; TMP.r.Set(x, z, y); bi.MoveKinematic(hand.GetID(), TMP.r, TMP.q, dt); };
      jw.release = () => { system.RemoveConstraint(con); bi.RemoveBody(hand.GetID()); bi.DestroyBody(hand.GetID()); };
      return jw;
    };
    W.release = g => { if (!g.grip) return; system.RemoveConstraint(g.grip.con); bi.RemoveBody(g.grip.hand.GetID()); bi.DestroyBody(g.grip.hand.GetID()); g.grip = null; };
    // put a glass somewhere outright (an observer's hand, or a move the bench does not model)
    W.placeGlass = (g, x, y, rot, z) => {
      const q = J.Quat.prototype.sRotation(v3(0, 1, 0), -rot);
      TMP.r.Set(x, z, y); bi.SetPositionAndRotation(g.id, TMP.r, q, J.EActivation_Activate);
      TMP.v.Set(0, 0, 0); bi.SetLinearVelocity(g.id, TMP.v); bi.SetAngularVelocity(g.id, TMP.v);
    };
    // a point fixed in a glass (its own frame, cm, y up from its underside) → bench coordinates, and back
    W.glassToWorld = (g, lx, ly, lz) => {
      const p = g.body.GetPosition(), r = g.body.GetRotation(), w = qrot(r.GetX(), r.GetY(), r.GetZ(), r.GetW(), lx, ly, lz);
      return { x: w[0] + p.GetX(), y: w[2] + p.GetZ(), z: w[1] + p.GetY() };
    };
    // for checking the check: how deep Jolt itself finds a ball (bench centre, radius) in a glass
    W.joltDepth = (g, b) => {
      const sh = new J.SphereShapeSettings(b.r).Create().Get(), p = toJ(b.x, b.y, b.z);
      const m = J.RMat44.prototype.sTranslation(r3(p[0], p[1], p[2])), cs = new J.CollideShapeSettings(), col = new J.CollideShapeAllHitCollisionCollector();
      cs.mMaxSeparationDistance = 0.05;
      g.body.GetTransformedShape().CollideShape(sh, v3(1, 1, 1), m, cs, r3(0, 0, 0), col, new J.ShapeFilter());
      let d = -Infinity; const hs = col.get_mHits(); for (let i = 0; i < hs.size(); i++) d = Math.max(d, hs.at(i).get_mPenetrationDepth());
      return d;
    };
    // how far the snout lags behind where the head would have it (cm, on the floor plan): held by
    // something, it is felt at once by the head it belongs to
    // how far the solver's body lags behind where the larva's own pose puts it (cm, the worst segment):
    // held back by something, it is felt
    // (the middle only: the tail swings in each stroke and always trails a little)
    W.bodyLag = (lv, all) => { let w = 0; for (const sg of lv.segs) if (sg.want && (all || !sg.isTail)) { const p = sg.body.GetPosition(); w = Math.max(w, Math.hypot(p.GetX() - sg.want[0], p.GetY() - sg.want[1], p.GetZ() - sg.want[2])); } return w; };
    W.snoutLag = lv => { const sn = lv.snout; if (!sn || !sn.want) return 0; const p = sn.body.GetPosition(); return Math.hypot(p.GetX() - sn.want.x, p.GetZ() - sn.want.y); };
    W.joltBodyHits = (g, body) => {
      const cs = new J.CollideShapeSettings(), col = new J.CollideShapeAllHitCollisionCollector(); cs.mMaxSeparationDistance = 0.0;
      g.body.GetTransformedShape().CollideShape(body.GetShape(), v3(1, 1, 1), body.GetCenterOfMassTransform(), cs, r3(0, 0, 0), col, new J.ShapeFilter());
      const out = [], hs = col.get_mHits();
      for (let i = 0; i < hs.size(); i++) { const h = hs.at(i), a = h.get_mContactPointOn1(), ax = a.GetX(), ay = a.GetY(), az = a.GetZ(), b = h.get_mContactPointOn2(), ax2 = h.get_mPenetrationAxis();
        out.push({ d: h.get_mPenetrationDepth(), on1: [ax, az, ay], on2: [b.GetX(), b.GetZ(), b.GetY()], axis: [ax2.GetX(), ax2.GetZ(), ax2.GetY()], sub2: h.get_mSubShapeID2().GetValue() }); }
      return out;
    };
    // the query objects are made once and reused: made at every call they filled the wasm heap (OOM)
    const JBD = {};
    W.joltBodyDepth = (g, body) => {
      if (!JBD.cs) { JBD.cs = new J.CollideShapeSettings(); JBD.cs.mMaxSeparationDistance = 0.05; JBD.sf = new J.ShapeFilter(); JBD.one = new J.Vec3(1, 1, 1); JBD.z = new J.RVec3(0, 0, 0); }
      const col = new J.CollideShapeAllHitCollisionCollector();
      g.body.GetTransformedShape().CollideShape(body.GetShape(), JBD.one, body.GetCenterOfMassTransform(), JBD.cs, JBD.z, col, JBD.sf);
      let d = -Infinity; const hs = col.get_mHits(); for (let i = 0; i < hs.size(); i++) d = Math.max(d, hs.at(i).get_mPenetrationDepth());
      J.destroy(col); return d;
    };
    W.worldToGlass = (g, q) => {
      const p = g.body.GetPosition(), r = g.body.GetRotation();
      return qrot(-r.GetX(), -r.GetY(), -r.GetZ(), r.GetW(), q.x - p.GetX(), q.z - p.GetY(), q.y - p.GetZ());
    };
    /* Where a glass is above a point of the floor: the upright line at (x, y) passes through
       the slab from height lo to hi (cm), or not at all (null). For drawing the slab cut
       exactly along the body, wherever the body lies. */
    W.glassSpan = (g, x, y) => {
      const a = W.worldToGlass(g, { x, y, z: 0 }), b = W.worldToGlass(g, { x, y, z: 1 });
      const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      if (Math.abs(d[1]) < 1e-6) return null;
      let z0 = (0 - a[1]) / d[1], z1 = (g.thick - a[1]) / d[1]; if (z0 > z1) [z0, z1] = [z1, z0];
      const inside = z => { const l = [a[0] + d[0] * z, a[1] + d[1] * z, a[2] + d[2] * z];
        const f = Math.max(0, Math.min(1, l[1] / g.thick)), top = g.localTop || g.local;
        let c = false; const P = g.local;
        for (let i = 0, j = P.length - 1; i < P.length; j = i++) {
          const pi = { x: P[i].x + (top[i].x - P[i].x) * f, y: P[i].y + (top[i].y - P[i].y) * f }, pj = { x: P[j].x + (top[j].x - P[j].x) * f, y: P[j].y + (top[j].y - P[j].y) * f };
          if ((pi.y > l[2]) !== (pj.y > l[2]) && l[0] < (pj.x - pi.x) * (l[2] - pi.y) / (pj.y - pi.y) + pi.x) c = !c; }
        return c; };
      let lo = null, hi = null;
      for (let k = 0; k <= 12; k++) { const z = z0 + (z1 - z0) * k / 12; if (inside(z)) { if (lo === null) lo = z; hi = z; } }
      return lo === null ? null : { lo, hi };
    };
    W.pointVelocity = (g, q) => { TMP.r.Set(q.x, q.z, q.y); const v = bi.GetPointVelocity(g.id, TMP.r); return { x: v.GetX(), y: v.GetZ(), z: v.GetY() }; };
    W.massOf = g => 1 / g.body.GetMotionProperties().GetInverseMass();
    W.glassSpeed = g => { const v = g.body.GetLinearVelocity(), a = g.body.GetAngularVelocity(); return { v: v.Length(), w: a.Length() }; };
    W.glassAngle = g => { const r = g.body.GetRotation(); const w = qrot(r.GetX(), r.GetY(), r.GetZ(), r.GetW(), 1, 0, 0); return Math.atan2(w[2], w[0]); };
    W.glassTilt = g => { const r = g.body.GetRotation(); const w = qrot(r.GetX(), r.GetY(), r.GetZ(), r.GetW(), 0, 1, 0); return Math.acos(Math.max(-1, Math.min(1, w[1]))); };

    /* A stone: a rigid ball on the floor. live: something alive (another larva)
       that the bench moves itself; nothing pushes it. */
    W.addBall = (x, y, r, live) => {
      // a pebble, round seen from above and flat-bottomed, so it slides and stops rather than rolling on for ever
      const hh = r * 0.6;
      const shape = new J.CylinderShapeSettings(hh, r, 0.03).Create().Get();
      const cs = new J.BodyCreationSettings(shape, r3(x, hh, y), J.Quat.prototype.sIdentity(), live ? J.EMotionType_Kinematic : J.EMotionType_Dynamic, LAYER_MOVING);
      cs.mFriction = 0.8; cs.mRestitution = 0; cs.mLinearDamping = 0.5; cs.mAngularDamping = 0.8;
      cs.mMotionQuality = J.EMotionQuality_LinearCast; cs.mAllowSleeping = false;
      if (!live) { cs.mOverrideMassProperties = J.EOverrideMassProperties_CalculateInertia; cs.mMassPropertiesOverride.mMass = 2.6 * Math.PI * r * r * 2 * hh; }
      const body = bi.CreateBody(cs); bi.AddBody(body.GetID(), J.EActivation_Activate); J.destroy(cs);
      const b = { body, id: body.GetID(), r, hh, live }; b.key = b.id.GetIndexAndSequenceNumber(); W.names.set(b.key, live ? 'the other larva' : 'a stone');
      W.stones.push(b); return b;
    };
    // a ball driven by its own muscles: toward a horizontal velocity, with at most this force
    W.driveBall = (b, vx, vy, maxF, dt) => {
      // the force that brings it to that speed in one step, plus what the floor's friction takes, at most maxF
      const v = b.body.GetLinearVelocity(), m = 1 / b.body.GetMotionProperties().GetInverseMass(), sp = Math.hypot(vx, vy) || 1;
      const fr = 0.8 * m * G * (Math.hypot(vx, vy) > 1e-6 ? 1 : 0);
      let fx = (vx - v.GetX()) * m / dt + vx / sp * fr, fz = (vy - v.GetZ()) * m / dt + vy / sp * fr; const f = Math.hypot(fx, fz);
      if (f > maxF) { fx *= maxF / f; fz *= maxF / f; }
      TMP.v.Set(fx, 0, fz); bi.AddForce(b.id, TMP.v, J.EActivation_Activate);
    };
    W.ballAt = b => { const p = b.body.GetPosition(); return { x: p.GetX(), y: p.GetZ(), z: p.GetY() }; };
    W.moveBall = (b, x, y, dt) => { TMP.r.Set(x, b.hh, y); bi.MoveKinematic(b.id, TMP.r, TMP.q, dt); };

    // the deepest that anything is inside anything else now (cm): rigid contacts,
    // and the skin's points inside a glass, a stone or the floor
    /* How deep anything is inside anything else NOW, measured on the shapes as
       the solver has left them (not on its contact reports, which are taken
       before it solves): glasses, the hard head, the soft segments, pebbles, the
       floor. Returns the deepest, in cm, and what it is. */
    const prismDist = (g, q) => {          // signed distance from a point to a glass slab (< 0 inside)
      const l = W.worldToGlass(g, q);
      // a bevelled piece: its outline at the point's height, between underside and top
      let poly = g.local;
      if (g.bevel) { const f = Math.max(0, Math.min(1, l[1] / g.thick)); poly = g.local.map((a, i) => ({ x: a.x + (g.localTop[i].x - a.x) * f, y: a.y + (g.localTop[i].y - a.y) * f })); }
      let inside = false, d = Infinity;
      for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const a = poly[j], b = poly[i];
        if (((b.y > l[2]) !== (a.y > l[2])) && (l[0] < (a.x - b.x) * (l[2] - b.y) / (a.y - b.y) + b.x)) inside = !inside;
        const dx = a.x - b.x, dy = a.y - b.y, t = Math.max(0, Math.min(1, ((l[0] - b.x) * dx + (l[2] - b.y) * dy) / (dx * dx + dy * dy)));
        d = Math.min(d, Math.hypot(l[0] - b.x - t * dx, l[2] - b.y - t * dy));
      }
      if (inside && l[1] > 0 && l[1] < g.thick) return -Math.min(d, l[1], g.thick - l[1]);
      const h = inside ? 0 : d, v = l[1] < 0 ? -l[1] : l[1] > g.thick ? l[1] - g.thick : 0;
      return Math.hypot(h, v);
    };
    // balls standing for each part of the larva: centre (bench coordinates) and radius
    const larvaBalls = lv => {
      const out = [];
      for (const sg of lv.segs) {
        // a tail segment is a 16-sided hull round its rings: its flat faces lie cos(π/16) of the
        // ring's radius from the axis (2 % in), and that, not the ring, is where it can be touched
        const p = sg.body.GetPosition(), r = sg.body.GetRotation(), rad = sg.isTail ? Math.min(lv.tailH(lv.FLAT[sg.flat]), lv.tailW(lv.FLAT[sg.flat])) * Math.cos(Math.PI / 16)
          : lv.soft ? Math.min(lv.midH(lv.FLAT[sg.flat]), lv.midW(lv.FLAT[sg.flat])) * Math.cos(Math.PI / 16) : sg.r;
        // at the two ends the round cap is cut in flat bands every 15°, which lie a further
        // cos(7.5°) in: the end balls are that much smaller, or they would stick out of the body
        const hull = sg.isTail || lv.soft, endK = hull ? Math.cos(7.5 * Math.PI / 180) : 1;
        for (let i = 0; i <= 4; i++) { const w = qrot(r.GetX(), r.GetY(), r.GetZ(), r.GetW(), 0, -sg.curLen / 2 + sg.curLen * i / 4, 0);
          out.push({ x: w[0] + p.GetX(), y: w[2] + p.GetZ(), z: w[1] + p.GetY(), r: rad * (i === 0 || i === 4 ? endK : 1), name: W.nameOf(sg.key) }); }
      }
      const h = W.headPoint(lv); out.push({ x: h.x, y: h.y, z: h.z, r: lv.headR, name: 'the head' });
      return out;
    };
    // the same moment is measured once (the bench asks twice a tick: to feel, and to check)
    let penAt = null, penWas = null;
    W.larvaBalls = larvaBalls;
    W.penetration = () => {
      if (penAt !== null && penAt.t === W.t && penAt.n === W.glasses.length) return penWas;
      let worst = 0, what = '';
      const hit = (d, w) => { if (d > worst) { worst = d; what = w; } };
      // each glass once: its corners, its middle and how far its farthest corner is from it,
      // so pairs that are far apart are skipped without measuring point by point
      const gl = W.glasses.map(g => {
        const pose = W.glassPose(g), pts = pose.bottom.concat(pose.top);
        let cx = 0, cy = 0, cz = 0; for (const q of pts) { cx += q.x; cy += q.y; cz += q.z; }
        cx /= pts.length; cy /= pts.length; cz /= pts.length;
        let R = 0; for (const q of pts) R = Math.max(R, Math.hypot(q.x - cx, q.y - cy, q.z - cz));
        return { g, pts, cx, cy, cz, R };
      });
      const near = (a, q, r) => Math.hypot(q.x - a.cx, q.y - a.cy, q.z - a.cz) <= a.R + r + 0.05;
      const balls = W.larvae.map(larvaBalls);
      for (const a of gl) {
        for (const q of a.pts) hit(-q.z, 'a glass in the floor');
        for (const b2 of gl) if (b2 !== a && Math.hypot(a.cx - b2.cx, a.cy - b2.cy, a.cz - b2.cz) <= a.R + b2.R + 0.05)
          for (const q of b2.pts) if (near(a, q, 0)) hit(-prismDist(a.g, q), 'a glass in a glass');
        for (const bs of balls) for (const b of bs) if (near(a, b, b.r)) hit(b.r - prismDist(a.g, b), b.name + ' in a glass');
        for (const st of W.stones) { const c = W.ballAt(st); if (near(a, c, st.r)) hit(st.r - prismDist(a.g, c), 'a stone in a glass'); }
      }
      for (const bs of balls) for (const b of bs) {
        hit(b.r - b.z, b.name + ' in the floor');
        for (const st of W.stones) { const c = W.ballAt(st), dz = Math.max(0, Math.abs(b.z - c.z) - st.hh), dh = Math.max(0, Math.hypot(b.x - c.x, b.y - c.y) - st.r);
          hit(b.r - Math.hypot(dz, dh), b.name + ' in ' + (st.live ? 'the other larva' : 'a stone')); }
      }
      for (const st of W.stones) { const c = W.ballAt(st); hit(st.hh - c.z, 'a stone in the floor');
        for (const s2 of W.stones) if (s2 !== st) { const c2 = W.ballAt(s2); if (Math.abs(c.z - c2.z) < st.hh + s2.hh) hit(st.r + s2.r - Math.hypot(c.x - c2.x, c.y - c2.y), 'a stone in a stone'); } }
      // the snout is no ball: Jolt itself is asked how deep it is in each glass near it
      for (const lv of W.larvae) if (lv.snout && lv.snout.placed) {
        const sb = lv.snout.body, c0 = sb.GetPosition(), c = { x: c0.GetX(), y: c0.GetY(), z: c0.GetZ() };
        hit(W.joltBodyDepth({ body: W.floor }, sb), 'the snout in the floor');
        for (const a of gl) if (Math.hypot(c.x - a.cx, c.z - a.cy, c.y - a.cz) <= a.R + 1) hit(W.joltBodyDepth(a.g, sb), 'the snout in a glass');
      }
      penAt = { t: W.t, n: W.glasses.length }; penWas = { depth: worst, what };
      return penWas;
    };
    // a segment caught between a heavy piece and the floor: now and then the solver, settling the
    // piece, leaves the segment a little in the floor. It is set back on the floor; if that puts it
    // into the piece, the sideways push-out that follows squeezes it out from under the edge
    const floorDepen = () => {
      for (const lv of W.larvae) { const bs = larvaBalls(lv);
        lv.segs.forEach((sg, i) => { let d = 0; for (let j = 5 * i; j < 5 * i + 5 && j < bs.length; j++) d = Math.max(d, bs[j].r - bs[j].z);
          if (d <= (W.tune.floorSlop || 0.005)) return;
          const b = sg.body, q = b.GetPosition(), np = new J.RVec3(q.GetX(), q.GetY() + d, q.GetZ());
          bi.SetPosition(b.GetID(), np, J.EActivation_Activate); J.destroy(np);
          const v = b.GetLinearVelocity(); if (v.GetY() < 0) { const nv = new J.Vec3(v.GetX(), 0, v.GetZ()); bi.SetLinearVelocity(b.GetID(), nv); J.destroy(nv); }
          W.floorDepenCount = (W.floorDepenCount || 0) + 1; W.floorDepenMax = Math.max(W.floorDepenMax || 0, d); }); }
    };
    const DEP = { cs: null, sf: null, one: null };
    const depen = () => {
      if (!DEP.cs) { DEP.cs = new J.CollideShapeSettings(); DEP.cs.mMaxSeparationDistance = 0; DEP.sf = new J.ShapeFilter(); DEP.one = new J.Vec3(1, 1, 1); DEP.z = new J.RVec3(0, 0, 0); }
      const slop = W.tune.depenSlop || 0.002;
      for (const lv of W.larvae) for (const sg of lv.segs) {
        const b = sg.body, p0 = b.GetPosition(), px = p0.GetX(), py = p0.GetY(), pz = p0.GetZ();
        for (let pass = 0; pass < 2; pass++) {
          let best = null;
          for (const g of W.glasses) { const gp = g.body.GetPosition(); if (Math.hypot(gp.GetX() - px, gp.GetY() - py, gp.GetZ() - pz) > (g.R || 6) + 2) continue;
            const col = new J.CollideShapeAllHitCollisionCollector();
            g.body.GetTransformedShape().CollideShape(b.GetShape(), DEP.one, b.GetCenterOfMassTransform(), DEP.cs, DEP.z, col, DEP.sf);
            const hs = col.get_mHits(); for (let i = 0; i < hs.size(); i++) { const h = hs.at(i), d = h.get_mPenetrationDepth(); if (d > slop && (!best || d > best.d)) { const a = h.get_mPenetrationAxis(); best = { d, ax: a.GetX(), ay: a.GetY(), az: a.GetZ(), g }; } }
            J.destroy(col); }
          if (!best) break;
          if (globalThis.__DBGD) console.log('DEPEN', W.t.toFixed(2), sg.isTail ? 'T' : 'M', lv.segs.indexOf(sg), pass, best.d.toFixed(4), best.ax.toFixed(2), best.ay.toFixed(2), best.az.toFixed(2));
          // (pressed down, between a glass and the floor, she does not sink into the floor for it: only
          // from the side is she pressed back out)
          // a slanting push (an edge resting on her side) moves her out sideways only, as far as it takes
          // to clear the glass along its push; a push straight down is left to the solver
          const al = Math.hypot(best.ax, best.ay, best.az) || 1, hl = Math.hypot(best.ax, best.az) / al;
          if (hl < (W.tune.depenSide || 0.35)) {
            // pressed straight down by a piece, with the floor under her, she does not give: the piece
            // is held up on her back (unless lifting it would only drive it into the piece above it)
            if (W.tune.liftGlass && best.ay * (W.tune.depenSign || -1) < 0 && best.g && best.g.body.GetMotionType() === J.EMotionType_Dynamic) {
              // the piece and whatever rests on it go up together (only by the sag the solver left: what
              // she can lift at all is the business of her behaviour, which backs out when strained)
              const S = [best.g]; for (let i = 0; i < S.length && S.length < 12; i++) { const s0 = S[i], y0 = s0.body.GetPosition().GetY();
                for (const o of W.glasses) if (!S.includes(o) && o.body.GetPosition().GetY() > y0 && Math.hypot(o.body.GetPosition().GetX() - s0.body.GetPosition().GetX(), o.body.GetPosition().GetZ() - s0.body.GetPosition().GetZ()) < (o.R || 6) + (s0.R || 6) && W.joltBodyDepth(o, s0.body) > -0.005) S.push(o); }
              const load = S.reduce((m, o) => m + W.massOf(o), 0);
              if (globalThis.__DBGD) console.log('LIFT', S.map(o => W.glasses.indexOf(o)).join('+'), 'load', load.toFixed(1), 'd', best.d.toFixed(4));
              if (!W.tune.maxLiftG || load <= W.tune.maxLiftG) {
                const rest = W.glasses.filter(o => !S.includes(o)), gg = () => { let m = 0; for (const x of S) for (const o of rest) { const xp = x.body.GetPosition(), op = o.body.GetPosition(); if (Math.hypot(xp.GetX() - op.GetX(), xp.GetZ() - op.GetZ()) < (x.R || 6) + (o.R || 6)) m = Math.max(m, W.joltBodyDepth(o, x.body)); } return m; };
                const before = gg(), was = S.map(o => { const q = o.body.GetPosition(); return [q.GetX(), q.GetY(), q.GetZ()]; });
                S.forEach((o, i) => { const up = new J.RVec3(was[i][0], was[i][1] + best.d, was[i][2]); bi.SetPosition(o.body.GetID(), up, J.EActivation_Activate); J.destroy(up); });
                if (globalThis.__DBGD) console.log('LIFT before', before.toFixed(4), 'after', gg().toFixed(4));
                if (gg() > Math.max(before, 0.003)) S.forEach((o, i) => { const bk = new J.RVec3(...was[i]); bi.SetPosition(o.body.GetID(), bk, J.EActivation_Activate); J.destroy(bk); });
                else { for (const o of S) { const gv = o.body.GetLinearVelocity(); if (gv.GetY() < 0) { const nv = new J.Vec3(gv.GetX(), 0, gv.GetZ()); bi.SetLinearVelocity(o.body.GetID(), nv); J.destroy(nv); } } W.liftCount = (W.liftCount || 0) + 1; } } }
            break; }
          const L = hl * al * (W.tune.depenSign || -1), ux = best.ax / L, uy = 0, uz = best.az / L, m = best.d / hl;
          // shape 2 (the segment) leaves along the penetration axis
          const q = b.GetPosition(), qx = q.GetX(), qy = q.GetY(), qz = q.GetZ(), np = new J.RVec3(qx + ux * m, qy + uy * m, qz + uz * m);
          bi.SetPosition(b.GetID(), np, J.EActivation_Activate); J.destroy(np);
          // (a push that drives her deeper — the axis Jolt gives for a segment caught in the corner between
          // two pieces can point into the other one — is taken back: she stays where the solver put her)
          if (W.tune.depenCheck) { let after = 0;
            for (const g of W.glasses) { const gp = g.body.GetPosition(); if (Math.hypot(gp.GetX() - px, gp.GetY() - py, gp.GetZ() - pz) > (g.R || 6) + 2) continue;
              const col = new J.CollideShapeAllHitCollisionCollector();
              g.body.GetTransformedShape().CollideShape(b.GetShape(), DEP.one, b.GetCenterOfMassTransform(), DEP.cs, DEP.z, col, DEP.sf);
              const hs = col.get_mHits(); for (let i = 0; i < hs.size(); i++) after = Math.max(after, hs.at(i).get_mPenetrationDepth()); J.destroy(col); }
            if (after >= best.d) { const bp = new J.RVec3(qx, qy, qz); bi.SetPosition(b.GetID(), bp, J.EActivation_Activate); J.destroy(bp); break; } }
          const v = b.GetLinearVelocity(), vn = v.GetX() * ux + v.GetY() * uy + v.GetZ() * uz;
          if (vn < 0) { const nv = new J.Vec3(v.GetX() - vn * ux, v.GetY() - vn * uy, v.GetZ() - vn * uz); bi.SetLinearVelocity(b.GetID(), nv); J.destroy(nv); }
          W.depenCount = (W.depenCount || 0) + 1; W.depenMax = Math.max(W.depenMax || 0, m);
        }
      }
    };
    W.step = dt => {
      // a piece tipping over or falling turns its corners fast: while one does, the step is cut
      // finer (W.tune.fastSubsteps), so a corner swinging down onto the floor does not sink in
      let n = W.tune.substeps;
      if (W.tune.fastSubsteps) for (const g of W.glasses) { const b = g.body; if (b.IsActive() && (b.GetAngularVelocity().Length() > 0.3 || b.GetLinearVelocity().Length() > 1)) { n = W.tune.fastSubsteps; break; } }
      W.contacts.length = 0; W.skinTouch.clear(); jolt.Step(dt, n); W.t += dt;
      // the larva is soft: where the solver let a segment of her sink into a glass this step
      // (a contact point it did not hold), her skin gives way and the segment is pressed back out
      // of it; the glass, far heavier, does not move for it
      // a tail segment (and, if soft, a middle one) pressed from above by something resting on
      // it flattens, step by step; with nothing on it, it rounds out again
      for (const lv of W.larvae) for (const sg of lv.segs) if (sg.isTail || lv.soft) {
        // something lying on it from above: a piece resting on a round body is seldom level, so a
        // slanting touch counts too
        const cz = sg.body.GetPosition().GetY();
        const above = W.contacts.filter(c => (c.a === sg.key || c.b === sg.key) && !W.isFloor(c.a === sg.key ? c.b : c.a) && !lv.keys.has(c.a === sg.key ? c.b : c.a) && c.at.z > cz);
        const pressed = above.some(c => Math.abs(c.n.z) > 0.3);
        // it flattens while pressed and rounds out again slowly, so a glass that rocks on it does not pump it
        // (with a piece still touching it from above it does not round out at all)
        sg.press = Math.max(0, Math.min(1, (sg.press || 0) + (pressed ? dt / W.tune.flatS : above.length ? 0 : -dt / W.tune.roundS)));
        // the larva can also make its tail thin itself, before it pushes in under something
        if (lv.thin && sg.isTail) sg.press = Math.min(1, sg.press + 2 * dt / W.tune.flatS);
        let want = Math.round(sg.press * (lv.FLAT.length - 1));
        // its length: the nearest step to what the larva draws it in to (the middle keeps its length)
        let wl = 0; if (sg.isTail) { for (let i = 0; i < lv.LENS.length; i++) if (Math.abs(lv.LENS[i] - lv.lenScale) < Math.abs(lv.LENS[wl] - lv.lenScale)) wl = i; }
        // it never swells or stretches into what lies on it: with a piece touching it from above it
        // keeps its shape until it is free (it can still flatten and draw in)
        if (above.length && (want < sg.flat || wl > sg.lenAt)) { want = Math.max(want, sg.flat); wl = Math.min(wl, sg.lenAt); }
        // and it changes shape one small step a tick, never at a jump
        want = Math.max(sg.flat - 1, Math.min(sg.flat + 1, want));
        if (want !== sg.flat || wl !== sg.lenAt) { sg.flat = want; sg.lenAt = wl; bi.SetShape(sg.body.GetID(), sg.shapes[want][wl], false, J.EActivation_Activate); if (W.tune.invCache !== 0) bi.InvalidateContactCache(sg.body.GetID()); }
      }
      // (after the tail has taken its new shape this step, so a shape change never leaves her in a glass)
      if (W.tune.floorDepen) floorDepen();
      if (W.tune.depen) depen();
    };
    W.isFloor = k => k === W.floorId;
    // contacts of one body this step, each with the other body's key
    W.touching = key => W.contacts.filter(c => c.a === key || c.b === key).map(c => ({ other: c.a === key ? c.b : c.a, depth: c.depth,
      n: c.a === key ? c.n : { x: -c.n.x, y: -c.n.y, z: -c.n.z }, at: c.at }));
    W.destroy = () => { J.destroy(jolt); };
    return W;
  }

  root.Physics = { load, createWorld, toJ, VR: 0 };
  if (typeof module !== 'undefined') module.exports = root.Physics;
})(typeof globalThis !== 'undefined' ? globalThis : this);
