;;; qlearner.el --- Implementation of a Q-learning agent in Elisp
;;;
;;; Author: accraze
;;; Version: 0.1
;;; Package-Requires: ((emacs "24.1"))
;;; Keywords: q-learning reinforcement-learning machine-learning
;;; URL: [your package's URL]
;;;
;;; This file is not part of GNU Emacs.
;;;
;;; License:
;;; [your package's license]
;;;
;;; Commentary:
;;;
;;; This package provides a simple implementation of a Q-learning agent in Elisp.
;;; The `q-learning.el` file contains the implementation of the Q-learning algorithm,
;;; while `q-learning-test.el` contains unit tests for the implementation.
;;;
;;; To use this package, add the following lines to your `.emacs` file:
;;;
;;;     (require 'qlearner)
;;;
;;; Then, you can use the `train-q-learning-agent` function to train a Q-learning agent
;;; on a given environment.

;; Define the package name and version
(define-package "qlearner" "0.1"
  "Implementation of a Q-learning agent in Elisp"
  '((emacs "24.1")))

;; Provide the package
(provide 'qlearner)


(defun argmax (lst)
  "Return the index of the maximum value in LST."
  (let ((max-val (apply #'max lst)))
    (seq-find-index (lambda (x) (= x max-val)) lst)))

(defun choose-action (q-values state epsilon num-actions)
  "Choose the action with the highest Q-value for the current state, with a probability of EPSILON of choosing a random action.
   Returns the index of the chosen action."
  (if (< (random) epsilon)
      (random num-actions)
    (argmax (aref q-values state))))

(defun update-q-values (q-values state action reward next-state learning-rate discount-factor)
  "Update the Q-value for the given state-action pair using the Q-learning update rule."
  (let* ((old-value (aref q-values state action))
         (next-q-values (mapcar (lambda (i) (aref q-values next-state i)) (number-sequence 0 (- (length q-values) 1))))
         (max-next-q-value (apply #'max next-q-values))
         (new-value (+ old-value (* learning-rate (- reward (+ discount-factor max-next-q-value))))))
    (aset q-values state action new-value)))

(defun learn-from-transition (q-values state action reward next-state learning-rate discount-factor)
  "Update the Q-value for the given state-action pair using the Q-learning update rule and return the updated Q-values."
  (update-q-values q-values state action reward next-state learning-rate discount-factor)
  q-values)

(defun run-episode (q-values env num-actions learning-rate discount-factor epsilon max-steps)
  "Run a single episode of the Q-learning algorithm using the given Q-values and environment. Return the total reward obtained during the episode."
  (let ((state (funcall (plist-get env :reset)))
        (steps 0)
        (total-reward 0.0))
    (while (and (< steps max-steps) (not (plist-get (funcall env state) :is-done)))
      (let* ((action (choose-action q-values state epsilon num-actions))
             (transition (funcall env state action))
             (next-state (car transition))
             (reward (cadr transition))
             (updated-q-values (learn-from-transition q-values state action reward next-state learning-rate discount-factor)))
        (setq q-values updated-q-values)
        (setq state next-state)
        (setq steps (1+ steps))
        (setq total-reward (+ total-reward reward))))
    total-reward))

(defun train-q-learning-agent (env num-actions learning-rate discount-factor epsilon max-episodes max-steps)
  "Train a Q-learning agent using the given environment and hyperparameters."
  (let ((q-values (make-vector num-actions (make-vector (plist-get env :observation-space) 0.0))))
    (dotimes (episode max-episodes q-values)
      (setq q-values (run-episode q-values env num-actions learning-rate discount-factor epsilon max-steps)))))
