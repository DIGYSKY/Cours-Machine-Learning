new_mail = {
    "emails": [
        # Emails originaux
        "You have won a free gift card",  # Spam
        "Vous avez gagné un cadeau gratuit",  # Spam
        "Meeting scheduled at 10am tomorrow",  # Non-spam
        "Earn money quickly with this secret method",  # Spam
        "Your bank account has been credited with $500.00",  # Non-spam
        "Limited offer, buy now",  # Spam
        "Can we reschedule our appointment?",  # Non-spam
        "Urgent! Your account has been compromised",  # Spam
        "Let's have lunch tomorrow",  # Non-spam
        "Votre commande a été expédiée",  # Non-spam
        # Nouveaux emails de test - Spam (English)
        "URGENT!!! Click here to claim your $10,000 prize NOW!!!",
        "Congratulations! You've been selected for a free vacation to Hawaii",
        "Make $5000 weekly from home. No experience needed. Start today!",
        "Your credit card will be blocked in 24 hours. Verify your account now",
        "Exclusive offer: Get 99% discount on all products. Limited time only!",
        "You have 1 unread message from a beautiful woman nearby",
        "Your device may be infected. Download our free antivirus immediately",
        "Claim your tax refund of $2,500. Click here to process",
        # Nouveaux emails de test - Spam (French)
        "URGENT!!! Cliquez ici pour réclamer votre prix de 10000€ MAINTENANT!!!",
        "Félicitations! Vous avez été sélectionné pour des vacances gratuites à Hawaii",
        "Gagnez 5000€ par semaine depuis chez vous. Aucune expérience requise. Commencez aujourd'hui!",
        "Votre carte de crédit sera bloquée dans 24 heures. Vérifiez votre compte maintenant",
        "Offre exclusive: Obtenez 99% de réduction sur tous les produits. Temps limité seulement!",
        "Vous avez 1 message non lu d'une belle femme à proximité",
        "Votre appareil peut être infecté. Téléchargez notre antivirus gratuit immédiatement",
        # Nouveaux emails de test - Non-spam (English)
        "Hi John, I wanted to follow up on our conversation from yesterday",
        "The quarterly business review meeting is scheduled for next Tuesday at 2 PM",
        "Thank you for submitting your application. We will review it and get back to you within 5 business days",
        "I've attached the project proposal document for your review. Please let me know your thoughts",
        "The software update has been successfully installed on all company computers",
        "Could you please send me the latest sales figures for Q4? I need them for the report",
        "I wanted to thank you for your excellent presentation at yesterday's meeting",
        "The client has approved the budget proposal. We can proceed with phase 2",
        "Hello team, the office will be closed next Monday for a public holiday",
        "I hope you're doing well. Would you like to schedule a coffee meeting next week?",
        # Nouveaux emails de test - Non-spam (French)
        "Bonjour Jean, je voulais faire un suivi sur notre conversation d'hier",
        "La réunion de révision trimestrielle est prévue pour mardi prochain à 14h",
        "Merci d'avoir soumis votre candidature. Nous l'examinerons et vous répondrons dans 5 jours ouvrables",
        "J'ai joint le document de proposition de projet pour votre examen. Faites-moi savoir vos pensées",
        "La mise à jour du logiciel a été installée avec succès sur tous les ordinateurs de l'entreprise",
        "Pourriez-vous m'envoyer les derniers chiffres de vente pour Q4? J'en ai besoin pour le rapport",
        "Je voulais vous remercier pour votre excellente présentation lors de la réunion d'hier",
        "Le client a approuvé la proposition budgétaire. Nous pouvons procéder à la phase 2",
        "Bonjour l'équipe, le bureau sera fermé lundi prochain pour un jour férié",
        "J'espère que vous allez bien. Souhaitez-vous planifier une réunion café la semaine prochaine?",
        # Cas limites - Ambigu (Non-spam mais peuvent sembler suspects)
        "Your package delivery is ready for pickup",
        "We have updated our privacy policy. Please review the changes",
        "Your subscription will renew automatically on the 15th of next month",
        # Plus de spam - English (10)
        "!!!WINNER!!! You've won $50,000!!! Click NOW to claim!!!",
        "Act fast! Your account expires in 12 hours. Renew immediately",
        "Get instant access to premium features for just $0.99. Limited offer!",
        "You're our 1,000,000th visitor! Claim your free iPhone now",
        "Warning: Suspicious activity detected. Verify your identity here",
        "Earn $200 daily by completing simple surveys. No investment needed",
        "Your computer performance is low. Download our optimization tool free",
        "Congratulations! You qualify for a $15,000 personal loan",
        "You have 5 unread messages from singles in your area. View now",
        "Your email storage is 99% full. Upgrade to premium for unlimited space",
        # Plus de spam - French (10)
        "!!!GAGNANT!!! Vous avez gagné 50000€!!! Cliquez MAINTENANT pour réclamer!!!",
        "Agissez vite! Votre compte expire dans 12 heures. Renouvelez immédiatement",
        "Obtenez un accès instantané aux fonctionnalités premium pour seulement 0,99€. Offre limitée!",
        "Vous êtes notre 1,000,000ème visiteur! Réclamez votre iPhone gratuit maintenant",
        "Avertissement: Activité suspecte détectée. Vérifiez votre identité ici",
        "Gagnez 200€ par jour en complétant des sondages simples. Aucun investissement requis",
        "Les performances de votre ordinateur sont faibles. Téléchargez notre outil d'optimisation gratuit",
        "Félicitations! Vous êtes éligible pour un prêt personnel de 15000€",
        "Vous avez 5 messages non lus de célibataires dans votre région. Voir maintenant",
        "Votre stockage email est plein à 99%. Passez à premium pour un espace illimité",
        # Plus de non-spam - English (12)
        "Good morning, I hope this email finds you well. I wanted to discuss the upcoming project deadline",
        "Thank you for your interest in our company. We have received your resume and will contact you soon",
        "The monthly team meeting has been rescheduled to Thursday at 3 PM due to scheduling conflicts",
        "I've completed the financial analysis you requested. The report is attached for your review",
        "Hello, I'm writing to confirm receipt of your payment. Thank you for your prompt response",
        "The training materials for the new software are now available in the shared drive",
        "I wanted to inform you that the office will be closed on Friday for a company event",
        "Could you please provide feedback on the proposal I sent last week? I'd appreciate your input",
        "The client has requested a meeting to discuss the project timeline. Are you available next Tuesday?",
        "I've reviewed your application and I'm pleased to inform you that you've been selected for an interview",
        "The quarterly sales report shows a 15% increase compared to last quarter. Great work team!",
        "Hello, I wanted to follow up on our discussion about the budget allocation for Q2",
        # Plus de non-spam - French (12)
        "Bonjour, j'espère que cet email vous trouve en bonne santé. Je voulais discuter de la date limite du projet à venir",
        "Merci pour votre intérêt pour notre entreprise. Nous avons reçu votre CV et vous contacterons bientôt",
        "La réunion d'équipe mensuelle a été reprogrammée à jeudi à 15h en raison de conflits d'horaires",
        "J'ai terminé l'analyse financière que vous avez demandée. Le rapport est joint pour votre examen",
        "Bonjour, j'écris pour confirmer la réception de votre paiement. Merci pour votre réponse rapide",
        "Les documents de formation pour le nouveau logiciel sont maintenant disponibles dans le dossier partagé",
        "Je voulais vous informer que le bureau sera fermé vendredi pour un événement d'entreprise",
        "Pourriez-vous fournir des commentaires sur la proposition que j'ai envoyée la semaine dernière? J'apprécierais votre avis",
        "Le client a demandé une réunion pour discuter du calendrier du projet. Êtes-vous disponible mardi prochain?",
        "J'ai examiné votre candidature et je suis heureux de vous informer que vous avez été sélectionné pour un entretien",
        "Le rapport de ventes trimestriel montre une augmentation de 15% par rapport au trimestre dernier. Excellent travail l'équipe!",
        "Bonjour, je voulais faire un suivi sur notre discussion concernant l'allocation budgétaire pour Q2",
        # Cas difficiles - Spam subtil (5)
        "Your order #12345 has been processed. Track your shipment at: bit.ly/track123",
        "We noticed unusual activity on your account. Please verify your login credentials",
        "Your subscription includes exclusive benefits. Upgrade to premium for only $9.99/month",
        "You have been pre-approved for our special offer. Click here to view details",
        "Your account needs attention. Please update your payment information to continue service",
        # Cas difficiles - Non-spam qui ressemble à du spam (5)
        "Your order confirmation: Order #789456 has been shipped. Expected delivery: 3-5 business days",
        "Security alert: We detected a login from a new device. If this was you, no action is needed",
        "Your subscription to our newsletter has been successfully activated. Thank you for joining!",
        "You have been selected to participate in our customer satisfaction survey. Your feedback is valuable",
        "Account update: Your payment method has been successfully updated. Thank you for keeping your information current"
    ],
    "isSpam": [
        # Labels originaux
        1,1,0,1,0,1,0,1,0,0,
        # Nouveaux spam - English (8)
        1,1,1,1,1,1,1,1,
        # Nouveaux spam - French (7)
        1,1,1,1,1,1,1,
        # Nouveaux non-spam - English (10)
        0,0,0,0,0,0,0,0,0,0,
        # Nouveaux non-spam - French (10)
        0,0,0,0,0,0,0,0,0,0,
        # Cas limites - Ambigu (3)
        0,0,0,
        # Plus de spam - English (10)
        1,1,1,1,1,1,1,1,1,1,
        # Plus de spam - French (10)
        1,1,1,1,1,1,1,1,1,1,
        # Plus de non-spam - English (12)
        0,0,0,0,0,0,0,0,0,0,0,0,
        # Plus de non-spam - French (12)
        0,0,0,0,0,0,0,0,0,0,0,0,
        # Cas difficiles - Spam subtil (5)
        1,1,1,1,1,
        # Cas difficiles - Non-spam qui ressemble à du spam (5)
        0,0,0,0,0
    ]
}
