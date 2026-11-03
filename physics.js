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
    ps.mSpeculativeContactDistance = 0.3;
    ps.mPenetrationSlop = 0.003;
    ps.mLinearCastThreshold = 0.5;
    ps.mManifoldTolerance = 0.05;
    ps.mNumVelocitySteps = 30;
    ps.mNumPositionSteps = 10;
    ps.mBaumgarte = 0.8;
    ps.mDeterministicSimulation = true;
    system.SetPhysicsSettings(ps);

    const W = { jolt, system, bi, glasses: [], stones: [], larvae: [], t: 0, names: new Map() };
    W.nameOf = k => W.names.get(k) || ('body ' + (k & 0x7fffff));
    // reused per step, so stepping allocates nothing in the solver's memory
    const TMP = { r: new J.RVec3(), v: new J.Vec3(), q: J.Quat.prototype.sIdentity(), r2: new J.Vec3() };
    W.tune = Object.assign({ springHz: 90, substeps: 8, segContact: 0.3, flatS: 0.6 }, opts.tune || {});
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
    const scale = (b1, b2, io) => {
      const a = J.wrapPointer(b1, J.Body).GetID().GetIndexAndSequenceNumber(), b = J.wrapPointer(b2, J.Body).GetID().GetIndexAndSequenceNumber();
      const st = J.wrapPointer(io, J.ContactSettings);
      for (const lv of W.larvae) {
        if (lv.segKeys.has(a)) { st.mInvMassScale1 = W.tune.segContact; st.mInvInertiaScale1 = W.tune.segContact; }
        if (lv.segKeys.has(b)) { st.mInvMassScale2 = W.tune.segContact; st.mInvInertiaScale2 = W.tune.segContact; }
      }
    };
    cl.OnContactAdded = (b1, b2, man, io) => { scale(b1, b2, io); rec(b1, b2, man); };
    cl.OnContactPersisted = (b1, b2, man, io) => { scale(b1, b2, io); rec(b1, b2, man); };
    cl.OnContactRemoved = () => {};
    system.SetContactListener(cl);

    // the floor
    {
      // no bigger than the bench: a huge box against small pieces loses precision
      const shape = new J.BoxShapeSettings(v3(20, 0.5, 15), 0).Create().Get();
      const cs = new J.BodyCreationSettings(shape, r3(16, -0.5, 10), J.Quat.prototype.sIdentity(), J.EMotionType_Static, LAYER_STATIC);
      cs.mFriction = 0.8; cs.mRestitution = 0;
      W.floor = bi.CreateBody(cs); bi.AddBody(W.floor.GetID(), J.EActivation_DontActivate);
      W.floorId = W.floor.GetID().GetIndexAndSequenceNumber(); W.names.set(W.floorId, 'the floor');
      J.destroy(cs);
    }

    /* A glass: a slab of the given outline (cm, in its own frame) and thickness. */
    W.addGlass = (local, thick, x, y, rot, z = 0) => {
      const hs = new J.ConvexHullShapeSettings();
      for (const q of local) { hs.mPoints.push_back(v3(q.x, 0, q.y)); hs.mPoints.push_back(v3(q.x, thick, q.y)); }
      hs.mMaxConvexRadius = 0.01; hs.mHullTolerance = 0.0005;
      hs.mDensity = GLASS_DENSITY;
      const res = hs.Create(); if (res.HasError()) throw new Error('glass hull: ' + res.GetError().c_str());
      const shape = res.Get();
      const q = J.Quat.prototype.sRotation(v3(0, 1, 0), -rot);   // bench rot is counter-clockwise seen from above
      const cs = new J.BodyCreationSettings(shape, r3(x, z, y), q, J.EMotionType_Dynamic, LAYER_MOVING);
      cs.mFriction = 0.6; cs.mRestitution = 0; cs.mLinearDamping = 0.2; cs.mAngularDamping = 0.4;
      cs.mMotionQuality = J.EMotionQuality_Discrete;
      cs.mUseManifoldReduction = false;     // keep every contact point: it rocks on its edges
      const body = bi.CreateBody(cs); bi.AddBody(body.GetID(), J.EActivation_Activate);
      J.destroy(cs);
      const g = { body, id: body.GetID(), local, thick };
      g.key = g.id.GetIndexAndSequenceNumber(); W.names.set(g.key, 'a glass');
      W.glasses.push(g);
      return g;
    };
    // where a glass is: its outline's points at the bottom and top, in bench coordinates
    W.glassPose = g => {
      const p = g.body.GetPosition(), r = g.body.GetRotation();
      const px = p.GetX(), py = p.GetY(), pz = p.GetZ(), qx = r.GetX(), qy = r.GetY(), qz = r.GetZ(), qw = r.GetW();
      const at = (lx, ly, lz) => { const w = qrot(qx, qy, qz, qw, lx, ly, lz); return { x: w[0] + px, y: w[2] + pz, z: w[1] + py }; };
      return { bottom: g.local.map(q => at(q.x, 0, q.y)), top: g.local.map(q => at(q.x, g.thick, q.y)) };
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
      const sausage = (len, h, w) => {
        const hs = new J.ConvexHullShapeSettings();
        const ring = (y, k) => { for (let i = 0; i < 16; i++) { const a = i / 16 * 2 * Math.PI; hs.mPoints.push_back(v3(Math.cos(a) * w * k, y, Math.sin(a) * h * k)); } };
        ring(-len / 2, 1); ring(len / 2, 1);
        for (const c of [30, 55, 75]) { const a = c * Math.PI / 180, r = Math.min(h, w); ring(-len / 2 - r * Math.sin(a), Math.cos(a)); ring(len / 2 + r * Math.sin(a), Math.cos(a)); }
        hs.mMaxConvexRadius = 0.01; hs.mHullTolerance = 0.0005;
        return hs.Create().Get();
      };
      // the soft tail, pressed from above, flattens: lower and wider, the same cross-section
      const FLAT = [0, 0.25, 0.5, 0.75, 1];
      const tailH = f => TAIL_D / 2 + (MID_D / 2 - TAIL_D / 2) * f, tailW = f => (TAIL_D / 2) * (TAIL_D / 2) / tailH(f);
      // segments: three on the tail, two on the middle; from the tail's end forward
      const cuts = [0, TAIL_LEN / 3, 2 * TAIL_LEN / 3, TAIL_LEN, TAIL_LEN + MID_LEN / 2, L];
      const massOf = (s0, s1) => (s1 <= TAIL_LEN ? 0.49 / TAIL_LEN : 0.23 / MID_LEN) * (s1 - s0);
      const segs = [];
      for (let i = 0; i + 1 < cuts.length; i++) {
        const s0 = cuts[i], s1 = cuts[i + 1], len = s1 - s0, isTail = s1 <= TAIL_LEN, r = isTail ? TAIL_D / 2 : MID_D / 2;
        // as long as the segment, its rounded ends overlapping the next one's
        const shapes = isTail ? FLAT.map(f => sausage(len, tailH(f), tailW(f))) : [new J.CapsuleShapeSettings(Math.max(0.01, len / 2), r).Create().Get()];
        const shape = shapes[0];
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
          const sp = c6.get_mLimitsSpringSettings(ax); sp.mFrequency = W.tune.springHz; sp.mDamping = 1;
        }
        const con = c6.Create(anchor, body); system.AddConstraint(con); J.destroy(c6);
        const seg = { body, anchor, s0, s1, r, len, isTail, shapes, flat: 0, key: body.GetID().GetIndexAndSequenceNumber(), placed: false };
        W.names.set(seg.key, s1 <= TAIL_LEN ? 'the tail' : 'the middle');
        segs.push(seg);
      }
      const hs = new J.SphereShapeSettings(HEAD_D / 2).Create().Get();
      const hcs = new J.BodyCreationSettings(hs, r3(0, 3, 0), J.Quat.prototype.sIdentity(), J.EMotionType_Kinematic, LAYER_LARVA);
      hcs.mFriction = 0.1; hcs.mRestitution = 0;    // smooth hard chitin
      const head = bi.CreateBody(hcs); bi.AddBody(head.GetID(), J.EActivation_Activate); J.destroy(hcs);
      // the height of the axis above the floor at s: the tail's is lower where it is flattened
      const radius = s => { for (const sg of segs) if (s <= sg.s1 + 1e-9) return sg.isTail ? tailH(FLAT[sg.flat]) : MID_D / 2; return MID_D / 2; };
      const larva = { segs, head, L, radius, tailH, tailW, FLAT, headR: HEAD_D / 2, headKey: head.GetID().GetIndexAndSequenceNumber(), segKeys: new Set(segs.map(q => q.key)) };
      larva.keys = new Set([...larva.segKeys, larva.headKey]);
      W.names.set(larva.headKey, 'the head');
      W.larvae.push(larva);
      return larva;
    };

    // a rotation taking Jolt's +Y (a capsule's axis) onto the direction d
    const quatYto = (dx, dy, dz) => {
      const n = Math.hypot(dx, dy, dz) || 1; dx /= n; dy /= n; dz /= n;
      const c = dy;                                   // cos of the angle between +Y and d
      if (c < -0.9999) return [1, 0, 0, 0];
      // axis = Y × d = (dz, 0, -dx)
      const ax = dz, az = -dx, s2 = Math.sqrt(2 * (1 + c));
      return [ax / s2, 0, az / s2, s2 / 2];
    };

    /* Move the skeleton. spine(s) gives, for arc length s from the tail's end,
       the axis point (bench coordinates, z up) and the direction along the body.
       head: the head's centre. Placed outright the first time, then moved over dt. */
    W.setPose = (larva, spine, headAt, dt) => {
      for (const sg of larva.segs) {
        // the segment lies along the spine's direction at its middle, its axis at the segment's own height
        const a = spine(sg.s0), b = spine(sg.s1), m = spine((sg.s0 + sg.s1) / 2);
        const pm = toJ(m.x, m.y, m.z), d = toJ(b.x - a.x, b.y - a.y, (b.z - a.z) - ((b.zr || 0) - (a.zr || 0)));
        const q = quatYto(d[0], d[1], d[2]);
        TMP.r.Set(pm[0], pm[1], pm[2]);
        const Q = new J.Quat(q[0], q[1], q[2], q[3]);
        if (sg.placed) bi.MoveKinematic(sg.anchor.GetID(), TMP.r, Q, dt);
        else {
          bi.SetPositionAndRotation(sg.anchor.GetID(), TMP.r, Q, J.EActivation_Activate);
          bi.SetPositionAndRotation(sg.body.GetID(), TMP.r, Q, J.EActivation_Activate);
          sg.placed = true;
        }
        J.destroy(Q);
      }
      const hp = toJ(headAt.x, headAt.y, headAt.z);
      TMP.r.Set(hp[0], hp[1], hp[2]);
      if (larva.headPlaced) bi.MoveKinematic(larva.head.GetID(), TMP.r, TMP.q, dt);
      else { bi.SetPosition(larva.head.GetID(), TMP.r, J.EActivation_Activate); larva.headPlaced = true; }
    };

    // the soft body's surface as points (bench coordinates): rings round each segment
    W.skinPoints = larva => {
      const out = [];
      for (const sg of larva.segs) {
        const p = sg.body.GetPosition(), r = sg.body.GetRotation(), qx = r.GetX(), qy = r.GetY(), qz = r.GetZ(), qw = r.GetW();
        const n = 4;
        for (let i = 0; i <= n; i++) {
          const y = -sg.len / 2 + sg.len * i / n;
          for (let k = 0; k < 10; k++) {
            const t = k / 10 * 2 * Math.PI, w = qrot(qx, qy, qz, qw, Math.cos(t) * sg.r, y, Math.sin(t) * sg.r);
            out.push({ x: w[0] + p.GetX(), y: w[2] + p.GetZ(), z: w[1] + p.GetY(), s: sg.s0 + (sg.s1 - sg.s0) * i / n, seg: sg.key });
          }
        }
      }
      return out;
    };
    // how far each segment has been pushed off its place on the skeleton (cm)
    W.give = larva => larva.segs.map(sg => { const a = sg.anchor.GetPosition(), b = sg.body.GetPosition(); return Math.hypot(a.GetX() - b.GetX(), a.GetY() - b.GetY(), a.GetZ() - b.GetZ()); });
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
    W.grip = g => {
      if (g.grip) return g.grip;
      const p = g.body.GetCenterOfMassPosition(), r = g.body.GetRotation();
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
      g.grip = { hand, con, x: p.GetX(), y: p.GetZ(), z: p.GetY(), yaw };
      return g.grip;
    };
    // move the gripping legs: the glass's centre to (x, y) and its turn to yaw, over dt
    W.moveGrip = (g, x, y, yaw, dt) => {
      const G = g.grip; G.x = x; G.y = y; G.yaw = yaw;
      TMP.r.Set(x, G.z, y);
      const q = J.Quat.prototype.sRotation(v3(0, 1, 0), -yaw);
      bi.MoveKinematic(G.hand.GetID(), TMP.r, q, dt);
    };
    // a hand lowering a glass: how much of its weight it carries itself (0 held, 1 let go)
    W.setWeight = (g, f) => bi.SetGravityFactor(g.id, f);
    /* Jaws: they take a glass at one point (fixed in the glass) and hold that point
       where they are moved to, through a stiff spring; the glass is free to turn
       about it. */
    W.jaws = (g, at) => {
      const as = new J.SphereShapeSettings(0.01).Create().Get();
      TMP.r.Set(at.x, at.z, at.y);
      const acs = new J.BodyCreationSettings(as, TMP.r, TMP.q, J.EMotionType_Kinematic, LAYER_GHOST);
      const hand = bi.CreateBody(acs); bi.AddBody(hand.GetID(), J.EActivation_Activate); J.destroy(acs);
      const c6 = new J.SixDOFConstraintSettings();
      c6.mSpace = J.EConstraintSpace_WorldSpace;
      c6.mPosition1 = r3(at.x, at.z, at.y); c6.mPosition2 = r3(at.x, at.z, at.y);
      // they hold that point firmly (through a stiff spring); the glass is free to turn about it
      for (const ax of [J.SixDOFConstraintSettings_EAxis_TranslationX, J.SixDOFConstraintSettings_EAxis_TranslationY, J.SixDOFConstraintSettings_EAxis_TranslationZ]) {
        c6.SetLimitedAxis(ax, 0, 0); const sp = c6.get_mLimitsSpringSettings(ax); sp.mFrequency = 20; sp.mDamping = 1;
      }
      for (const ax of [J.SixDOFConstraintSettings_EAxis_RotationX, J.SixDOFConstraintSettings_EAxis_RotationY, J.SixDOFConstraintSettings_EAxis_RotationZ]) c6.MakeFreeAxis(ax);
      const con = c6.Create(hand, g.body); system.AddConstraint(con); J.destroy(c6);
      const jw = { hand, con, x: at.x, y: at.y, z: at.z };
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
    W.worldToGlass = (g, q) => {
      const p = g.body.GetPosition(), r = g.body.GetRotation();
      return qrot(-r.GetX(), -r.GetY(), -r.GetZ(), r.GetW(), q.x - p.GetX(), q.z - p.GetY(), q.y - p.GetZ());
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
      cs.mMotionQuality = J.EMotionQuality_LinearCast;
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
      const l = W.worldToGlass(g, q), poly = g.local; let inside = false, d = Infinity;
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
        const p = sg.body.GetPosition(), r = sg.body.GetRotation(), rad = sg.isTail ? Math.min(lv.tailH(lv.FLAT[sg.flat]), lv.tailW(lv.FLAT[sg.flat])) : sg.r;
        for (let i = 0; i <= 4; i++) { const w = qrot(r.GetX(), r.GetY(), r.GetZ(), r.GetW(), 0, -sg.len / 2 + sg.len * i / 4, 0);
          out.push({ x: w[0] + p.GetX(), y: w[2] + p.GetZ(), z: w[1] + p.GetY(), r: rad, name: W.nameOf(sg.key) }); }
      }
      const h = W.headPoint(lv); out.push({ x: h.x, y: h.y, z: h.z, r: lv.headR, name: 'the head' });
      return out;
    };
    W.penetration = () => {
      let worst = 0, what = '';
      const hit = (d, w) => { if (d > worst) { worst = d; what = w; } };
      for (const g of W.glasses) {
        const pose = W.glassPose(g);
        for (const q of pose.bottom.concat(pose.top)) hit(-q.z, 'a glass in the floor');
        for (const g2 of W.glasses) if (g2 !== g) { const p2 = W.glassPose(g2); for (const q of p2.bottom.concat(p2.top)) hit(-prismDist(g, q), 'a glass in a glass'); }
        for (const lv of W.larvae) for (const b of larvaBalls(lv)) hit(b.r - prismDist(g, b), b.name + ' in a glass');
        for (const st of W.stones) { const c = W.ballAt(st); hit(st.r - prismDist(g, c), 'a stone in a glass'); }
      }
      for (const lv of W.larvae) for (const b of larvaBalls(lv)) {
        hit(b.r - b.z, b.name + ' in the floor');
        for (const st of W.stones) { const c = W.ballAt(st), dz = Math.max(0, Math.abs(b.z - c.z) - st.hh), dh = Math.max(0, Math.hypot(b.x - c.x, b.y - c.y) - st.r);
          hit(b.r - Math.hypot(dz, dh), b.name + ' in ' + (st.live ? 'the other larva' : 'a stone')); }
      }
      for (const st of W.stones) { const c = W.ballAt(st); hit(st.hh - c.z, 'a stone in the floor');
        for (const s2 of W.stones) if (s2 !== st) { const c2 = W.ballAt(s2); if (Math.abs(c.z - c2.z) < st.hh + s2.hh) hit(st.r + s2.r - Math.hypot(c.x - c2.x, c.y - c2.y), 'a stone in a stone'); } }
      return { depth: worst, what };
    };
    W.step = dt => {
      W.contacts.length = 0; W.skinTouch.clear(); jolt.Step(dt, W.tune.substeps); W.t += dt;
      // a tail segment pressed from above by something resting on it flattens, step by
      // step; with nothing on it, it rounds out again
      for (const lv of W.larvae) for (const sg of lv.segs) if (sg.isTail) {
        const pressed = W.contacts.some(c => (c.a === sg.key || c.b === sg.key) && !W.isFloor(c.a === sg.key ? c.b : c.a) && !lv.keys.has(c.a === sg.key ? c.b : c.a) && Math.abs(c.n.z) > 0.7 && c.at.z > (sg.body.GetPosition().GetY()));
        sg.press = Math.max(0, Math.min(1, (sg.press || 0) + (pressed ? dt : -dt) / W.tune.flatS));
        const want = Math.round(sg.press * (lv.FLAT.length - 1));
        if (want !== sg.flat) { sg.flat = want; bi.SetShape(sg.body.GetID(), sg.shapes[want], false, J.EActivation_Activate); }
      }
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
