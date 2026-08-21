(set-logic QF_NRA)
; Parameters
(declare-const phi_0 Real) ; Initial angle
(declare-const phi_u_lower Real) ; Unsafe set lower bound
(declare-const phi_u_upper Real) ; Unsafe set upper bound
(declare-const phi_u Real) ; Unsafe angle in [q_u, w_u]
(declare-const omega Real)
(declare-const theta_b Real)
(declare-const delta Real)
(declare-const M Real)
(declare-const N Real)
(declare-const T Int) ; Integer horizon
(declare-const t Int)
(declare-const M_rumoroso Real)

(assert (= T 2))
(assert (and (>= t 0) (< t T)))
(assert (= delta 0))

; Grover's parameters (N=4, M=1, err=0.5)
(assert (= N 8.0))
;(assert (= M 1.0))
(assert (and (>= M 0.5) (<= M 1.5)))
(assert (and (>= M_rumoroso 0.5) (<= M_rumoroso 1.5)))

; Grover's rotation angle theta
(define-fun theta_g () Real (* 2.0 (arcsin (sqrt (/ M N)))))
(define-fun phi_g () Real (* 2.0 (arcsin (sqrt (/ M_rumoroso N)))))

; Barrier function cos()
(define-fun B ((phi Real)) Real (cos phi))

; Initial state
(assert (= phi_0 (arcsin (sqrt (/ M N))))) ; Initial angle (phi_0)

; Unsafe set bounds (Xᵤ = [phiᵤ_low, phiᵤ_up])
(assert (= phi_u_lower 0.0)) ; angles near |α⟩
(assert (= phi_u_upper (arcsin (sqrt (/ 0.4 N))))) ; Threshold for M=0.4
(assert (and (<= phi_u phi_u_upper) (>= phi_u phi_u_lower)))

; Constraint B(phi₀) ≤ ω initial states
(assert (= omega (cos (arcsin (sqrt (/ 0.5 N))))))
(assert (> (B phi_0) omega))

; Constraint B(phiᵤ) > θ_b unsafe states
(assert (= theta_b (+ (cos phi_u_upper) 0.0001)))
(assert (<= (B phi_u) theta_b))

; dynamics for T steps
(define-fun phi_t ((t Int)) Real (+ phi_0 (* t phi_g)))

; B(phi_{t+1}) - B(phi_t) ≤ δ for t=0,...,T-1
;(assert  (> (- (B (phi_t (+ t 1))) (B (phi_t t))) delta))
(assert (forall ((t Int)) (=> (and (>= t 0) (< t T))(> (- (B (phi_t (+ t 1))) (B (phi_t t))) delta))))

; ω + T*δ < θ_b
(assert (< (+ omega (* T delta)) theta_b))
(assert (< omega theta_b))
(assert (>= theta_b 0))

;(assert (not (and (>= (phi_t t) q_u) (<= (phi_t t) w_u))))

(check-sat)