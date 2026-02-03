"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronLeft, ChevronRight, Check, Loader2, Eye, EyeOff, Github, Mail, Lock, User, Target, Briefcase } from "lucide-react";
import { Button } from "./ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { RadioGroup, RadioGroupItem } from "./ui/radio-group";
import { Textarea } from "./ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { cn } from "../lib/utils";
import { useAuth } from "../lib/AuthContext";
import { Logo } from "./Logo";
import { saveUserProfile } from "../lib/supabase";
import { supabase } from "../lib/supabase";

const steps = [
  { id: "personal", title: "Personal Info" },
  { id: "goals", title: "Data Science Goals" },
  { id: "professional", title: "Professional" },
];

interface FormData {
  name: string;
  email: string;
  password: string;
  confirmPassword: string;
  primaryGoal: string;
  targetOutcome: string;
  dataTypes: string[];
  profession: string;
  experience: string;
  industry: string;
}

const fadeInUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.3 } },
};

const contentVariants = {
  hidden: { opacity: 0, x: 50 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.3 } },
  exit: { opacity: 0, x: -50, transition: { duration: 0.2 } },
};

interface AuthPageProps {
  onSuccess?: () => void;
  onSkip?: () => void;
}

export const AuthPage: React.FC<AuthPageProps> = ({ onSuccess, onSkip }) => {
  const { signIn, signUp, signInWithGoogle, signInWithGithub, isConfigured, user, refreshOnboardingStatus } = useAuth();
  const [mode, setMode] = useState<'signin' | 'signup'>('signin');
  const [currentStep, setCurrentStep] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  const [formData, setFormData] = useState<FormData>({
    name: "",
    email: "",
    password: "",
    confirmPassword: "",
    primaryGoal: "",
    targetOutcome: "",
    dataTypes: [],
    profession: "",
    experience: "",
    industry: "",
  });

  // If user is already authenticated (OAuth), pre-fill email and switch to signup mode for onboarding
  React.useEffect(() => {
    if (user && user.email) {
      setFormData(prev => ({
        ...prev,
        email: user.email || '',
        name: user.user_metadata?.full_name || user.user_metadata?.name || ''
      }));
      setMode('signup');
    }
  }, [user]);

  const updateFormData = (field: keyof FormData, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    setError(null);
  };

  const nextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep((prev) => prev + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep((prev) => prev - 1);
    }
  };

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!isConfigured) {
      setError('Authentication is not configured. Please contact the administrator.');
      return;
    }
    
    setIsSubmitting(true);
    setError(null);

    try {
      const { error } = await signIn(formData.email, formData.password);
      if (error) {
        setError(error.message);
      } else {
        onSuccess?.();
      }
    } catch (err: any) {
      if (err.message?.includes('Failed to fetch')) {
        setError('Unable to connect to authentication server. Please try again later.');
      } else {
        setError(err.message || 'An error occurred');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleSignUp = async () => {
    if (!isConfigured) {
      setError('Authentication is not configured. Please contact the administrator.');
      return;
    }
    
    setIsSubmitting(true);
    setError(null);

    // Check if user is already authenticated (OAuth flow)
    const isOAuthUser = !!user;
    console.log('handleSignUp called, isOAuthUser:', isOAuthUser, 'user:', user);
    
    try {
      let userId: string;
      let userEmail: string;
      
      if (isOAuthUser) {
        // User already authenticated via OAuth, just save profile
        userId = user.id;
        userEmail = user.email || formData.email;
        console.log('OAuth user detected, saving profile only. userId:', userId);
      } else {
        // Email/password signup
        if (formData.password !== formData.confirmPassword) {
          setError("Passwords don't match");
          setIsSubmitting(false);
          return;
        }

        console.log('Email/password signup, creating new account...');
        const { error } = await signUp(formData.email, formData.password);
        if (error) {
          console.error('Signup error:', error);
          setError(error.message);
          setIsSubmitting(false);
          return;
        }
        
        // Wait for Supabase to create the auth user
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Get the user ID from auth session
        const { data: { session } } = await supabase.auth.getSession();
        if (!session?.user) {
          setError('Failed to get user session. Please sign in to continue.');
          setIsSubmitting(false);
          return;
        }
        userId = session.user.id;
        userEmail = session.user.email || formData.email;
        console.log('New account created, userId:', userId);
      }
      
      // Save user profile data to database (no HF token in signup)
      const profileData = {
        user_id: userId,
        name: formData.name,
        email: userEmail,
        primary_goal: formData.primaryGoal,
        target_outcome: formData.targetOutcome,
        data_types: formData.dataTypes,
        profession: formData.profession,
        experience: formData.experience,
        industry: formData.industry,
        onboarding_completed: true
      };
      
      console.log('Saving profile data:', profileData);
      
      // Add timeout to prevent infinite hanging
      const saveProfileWithTimeout = Promise.race([
        saveUserProfile(profileData),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Profile save timeout')), 10000)
        )
      ]);
      
      try {
        const savedProfile = await saveProfileWithTimeout;
        if (!savedProfile) {
          console.error('Failed to save profile data');
          setError('Failed to save your profile. Please try again.');
          setIsSubmitting(false);
          return;
        }
        
        console.log('Profile saved successfully:', savedProfile);
        
        // Refresh onboarding status so AuthContext knows the profile is complete
        await refreshOnboardingStatus();
        
        // Only proceed if profile was saved successfully
        setSuccess(isOAuthUser ? 'Profile completed! Redirecting...' : 'Account created successfully! Redirecting...');
        setTimeout(() => {
          onSuccess?.();
        }, 1500);
      } catch (saveError: any) {
        console.error('Profile save error:', saveError);
        if (saveError.message === 'Profile save timeout') {
          setError('Profile save is taking too long. Please try again.');
        } else {
          setError('Failed to save your profile. Please try again.');
        }
        setIsSubmitting(false);
        return;
      }
    } catch (err: any) {
      if (err.message?.includes('Failed to fetch')) {
        setError('Unable to connect to authentication server. Please try again later.');
      } else {
        setError(err.message || 'An error occurred');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleOAuthSignIn = async (provider: 'google' | 'github') => {
    if (!isConfigured) {
      setError('Authentication is not configured. Please contact the administrator.');
      return;
    }
    
    setIsSubmitting(true);
    setError(null);

    try {
      const { error } = provider === 'google' ? await signInWithGoogle() : await signInWithGithub();
      if (error) {
        setError(error.message);
      }
    } catch (err: any) {
      if (err.message?.includes('Failed to fetch')) {
        setError('Unable to connect to authentication server. Please try again later.');
      } else {
        setError(err.message || 'An error occurred');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const isStepValid = () => {
    // For OAuth users (already authenticated), skip password validation
    const isOAuthUser = !!user;
    
    switch (currentStep) {
      case 0:
        if (isOAuthUser) {
          // OAuth users don't need password fields
          return formData.name.trim() !== "" && formData.email.trim() !== "";
        }
        return formData.name.trim() !== "" && formData.email.trim() !== "" && 
               formData.password.length >= 6 && formData.password === formData.confirmPassword;
      case 1:
        return formData.primaryGoal !== "";
      case 2:
        return formData.profession.trim() !== "" && formData.industry !== "";
      case 3:
        // HuggingFace token is optional, always valid
        return true;
      default:
        return true;
    }
  };

  // Sign In Form
  if (mode === 'signin') {
    return (
      <div className="min-h-screen bg-[#030303] flex items-center justify-center p-4">
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-indigo-500/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-500/10 rounded-full blur-3xl" />
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative w-full max-w-md"
        >
          <Card className="border-white/10 bg-white/[0.03] backdrop-blur-xl shadow-2xl pb-0">
            <CardHeader className="space-y-1 text-center mb-2 mt-4">
              <div className="flex justify-center mb-2">
                <Logo className="w-12 h-12" />
              </div>
              <div>
                <h2 className="text-2xl font-semibold text-white">Sign in to Data Science Agent</h2>
                <p className="text-white/50 text-sm mt-1">
                  Welcome back! Please enter your details.
                </p>
              </div>
            </CardHeader>
            <CardContent className="space-y-4 px-8">
              <div className="space-y-3">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => handleOAuthSignIn('google')}
                  disabled={isSubmitting}
                  className="w-full bg-white/5 hover:bg-white/10 border-white/10 text-white h-11"
                >
                  <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24">
                    <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                  </svg>
                  Continue with Google
                </Button>

                <Button
                  type="button"
                  variant="outline"
                  onClick={() => handleOAuthSignIn('github')}
                  disabled={isSubmitting}
                  className="w-full bg-white/5 hover:bg-white/10 border-white/10 text-white h-11"
                >
                  <Github className="w-5 h-5 mr-2" />
                  Continue with GitHub
                </Button>
              </div>

              <div className="relative my-6">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-white/10"></div>
                </div>
                <div className="relative flex justify-center text-sm">
                  <span className="px-4 bg-[#030303] text-white/40">or continue with email</span>
                </div>
              </div>

              <form onSubmit={handleSignIn} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email" className="text-white/70">Email address</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30" />
                    <Input
                      id="email"
                      type="email"
                      placeholder="you@example.com"
                      value={formData.email}
                      onChange={(e) => updateFormData("email", e.target.value)}
                      className="pl-10 bg-white/5 border-white/10 text-white placeholder:text-white/30 focus:border-indigo-500/50"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="password" className="text-white/70">Password</Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30" />
                    <Input
                      id="password"
                      type={showPassword ? "text" : "password"}
                      placeholder="••••••••"
                      value={formData.password}
                      onChange={(e) => updateFormData("password", e.target.value)}
                      className="pl-10 pr-10 bg-white/5 border-white/10 text-white placeholder:text-white/30 focus:border-indigo-500/50"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-white/30 hover:text-white/50"
                    >
                      {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                <AnimatePresence>
                  {error && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm"
                    >
                      {error}
                    </motion.div>
                  )}
                </AnimatePresence>

                <Button
                  type="submit"
                  disabled={isSubmitting || !formData.email || !formData.password}
                  className="w-full bg-indigo-600 hover:bg-indigo-700 text-white h-11"
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Signing in...
                    </>
                  ) : (
                    "Sign In"
                  )}
                </Button>
              </form>

              <Button
                type="button"
                variant="ghost"
                onClick={onSkip}
                className="w-full text-white/50 hover:text-white hover:bg-white/5"
              >
                Continue as Guest
              </Button>
            </CardContent>
            <CardFooter className="flex justify-center border-t border-white/10 py-4">
              <p className="text-center text-sm text-white/50">
                New to Data Science Agent?{" "}
                <button
                  onClick={() => {
                    setMode('signup');
                    setError(null);
                    setCurrentStep(0);
                  }}
                  className="text-indigo-400 hover:text-indigo-300 hover:underline"
                >
                  Sign up
                </button>
              </p>
            </CardFooter>
          </Card>
        </motion.div>
      </div>
    );
  }

  // Sign Up Multi-Step Form
  return (
    <div className="min-h-screen bg-[#030303] flex items-center justify-center p-4">
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-indigo-500/10 rounded-full blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-500/10 rounded-full blur-3xl" />
      </div>

      <div className="relative w-full max-w-lg py-8">
        <motion.div
          className="mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="flex justify-between mb-2">
            {steps.map((step, index) => (
              <motion.div
                key={index}
                className="flex flex-col items-center"
                whileHover={{ scale: 1.1 }}
              >
                <motion.div
                  className={cn(
                    "w-4 h-4 rounded-full cursor-pointer transition-colors duration-300",
                    index < currentStep
                      ? "bg-indigo-500"
                      : index === currentStep
                        ? "bg-indigo-500 ring-4 ring-indigo-500/20"
                        : "bg-white/20",
                  )}
                  onClick={() => {
                    if (index <= currentStep) {
                      setCurrentStep(index);
                    }
                  }}
                  whileTap={{ scale: 0.95 }}
                />
                <motion.span
                  className={cn(
                    "text-xs mt-1.5 hidden sm:block",
                    index === currentStep
                      ? "text-indigo-400 font-medium"
                      : "text-white/40",
                  )}
                >
                  {step.title}
                </motion.span>
              </motion.div>
            ))}
          </div>
          <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden mt-2">
            <motion.div
              className="h-full bg-indigo-500"
              initial={{ width: 0 }}
              animate={{ width: `${(currentStep / (steps.length - 1)) * 100}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Card className="border-white/10 bg-white/[0.03] backdrop-blur-xl shadow-2xl overflow-hidden">
            <div>
              <AnimatePresence mode="wait">
                <motion.div
                  key={currentStep}
                  initial="hidden"
                  animate="visible"
                  exit="exit"
                  variants={contentVariants}
                >
                  {currentStep === 0 && (
                    <>
                      <CardHeader>
                        <div className="flex justify-center mb-2">
                          <Logo className="w-10 h-10" />
                        </div>
                        <CardTitle className="text-white text-center">Create your account</CardTitle>
                        <CardDescription className="text-white/50 text-center">
                          Let's start with some basic information
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <motion.div variants={fadeInUp} className="space-y-2">
                          <Label htmlFor="name" className="text-white/70">Full Name</Label>
                          <div className="relative">
                            <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30" />
                            <Input
                              id="name"
                              placeholder="John Doe"
                              value={formData.name}
                              onChange={(e) => updateFormData("name", e.target.value)}
                              className="pl-10 bg-white/5 border-white/10 text-white placeholder:text-white/30 focus:border-indigo-500/50"
                            />
                          </div>
                        </motion.div>
                        <motion.div variants={fadeInUp} className="space-y-2">
                          <Label htmlFor="signup-email" className="text-white/70">Email Address</Label>
                          <div className="relative">
                            <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30" />
                            <Input
                              id="signup-email"
                              type="email"
                              placeholder="john@example.com"
                              value={formData.email}
                              onChange={(e) => updateFormData("email", e.target.value)}
                              disabled={!!user}
                              className={cn(
                                "pl-10 bg-white/5 border-white/10 text-white placeholder:text-white/30 focus:border-indigo-500/50",
                                user && "opacity-60 cursor-not-allowed"
                              )}
                            />
                            {user && (
                              <p className="text-xs text-white/40 mt-1">Email from OAuth provider</p>
                            )}
                          </div>
                        </motion.div>
                        
                        {/* Only show password fields for email/password signup (not OAuth) */}
                        {!user && (
                          <>
                            <motion.div variants={fadeInUp} className="space-y-2">
                              <Label htmlFor="signup-password" className="text-white/70">Password</Label>
                              <div className="relative">
                                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30" />
                                <Input
                                  id="signup-password"
                                  type={showPassword ? "text" : "password"}
                                  placeholder="••••••••"
                                  value={formData.password}
                                  onChange={(e) => updateFormData("password", e.target.value)}
                                  className="pl-10 pr-10 bg-white/5 border-white/10 text-white placeholder:text-white/30 focus:border-indigo-500/50"
                                />
                                <button
                                  type="button"
                                  onClick={() => setShowPassword(!showPassword)}
                                  className="absolute right-3 top-1/2 -translate-y-1/2 text-white/30 hover:text-white/50"
                                >
                                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                </button>
                              </div>
                              <p className="text-xs text-white/40">Minimum 6 characters</p>
                            </motion.div>
                            <motion.div variants={fadeInUp} className="space-y-2">
                              <Label htmlFor="confirm-password" className="text-white/70">Confirm Password</Label>
                              <div className="relative">
                                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30" />
                                <Input
                                  id="confirm-password"
                                  type={showPassword ? "text" : "password"}
                                  placeholder="••••••••"
                                  value={formData.confirmPassword}
                                  onChange={(e) => updateFormData("confirmPassword", e.target.value)}
                                  className={cn(
                                    "pl-10 bg-white/5 border-white/10 text-white placeholder:text-white/30 focus:border-indigo-500/50",
                                    formData.confirmPassword && formData.password !== formData.confirmPassword && "border-red-500/50"
                                  )}
                                />
                              </div>
                              {formData.confirmPassword && formData.password !== formData.confirmPassword && (
                                <p className="text-xs text-red-400">Passwords don't match</p>
                              )}
                            </motion.div>
                          </>
                        )}

                        {/* Only show OAuth buttons for non-authenticated users */}
                        {!user && (
                          <>
                            <div className="relative my-4">
                              <div className="absolute inset-0 flex items-center">
                                <div className="w-full border-t border-white/10"></div>
                              </div>
                              <div className="relative flex justify-center text-sm">
                                <span className="px-4 bg-[#0a0a0a] text-white/40">or sign up with</span>
                              </div>
                            </div>

                        <div className="flex gap-3">
                          <Button
                            type="button"
                            variant="outline"
                            onClick={() => handleOAuthSignIn('google')}
                            disabled={isSubmitting}
                            className="flex-1 bg-white/5 hover:bg-white/10 border-white/10 text-white"
                          >
                            <svg className="w-4 h-4 mr-2" viewBox="0 0 24 24">
                              <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                              <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                              <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                              <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                            </svg>
                            Google
                          </Button>
                          <Button
                            type="button"
                            variant="outline"
                            onClick={() => handleOAuthSignIn('github')}
                            disabled={isSubmitting}
                            className="flex-1 bg-white/5 hover:bg-white/10 border-white/10 text-white"
                          >
                            <Github className="w-4 h-4 mr-2" />
                            GitHub
                          </Button>
                        </div>
                          </>
                        )}
                      </CardContent>
                    </>
                  )}

                  {currentStep === 1 && (
                    <>
                      <CardHeader>
                        <div className="flex justify-center mb-2">
                          <div className="w-12 h-12 bg-indigo-500/10 rounded-full flex items-center justify-center">
                            <Target className="w-6 h-6 text-indigo-400" />
                          </div>
                        </div>
                        <CardTitle className="text-white text-center">Data Science Goals</CardTitle>
                        <CardDescription className="text-white/50 text-center">
                          What are you trying to achieve with data science?
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <motion.div variants={fadeInUp} className="space-y-2">
                          <Label className="text-white/70">What's your primary goal?</Label>
                          <RadioGroup
                            value={formData.primaryGoal}
                            onValueChange={(value) => updateFormData("primaryGoal", value)}
                            className="space-y-2"
                          >
                            {[
                              { value: "explore-data", label: "Explore and understand my data", icon: "🔍" },
                              { value: "build-models", label: "Build predictive models", icon: "🤖" },
                              { value: "automate-analysis", label: "Automate data analysis workflows", icon: "⚡" },
                              { value: "visualize", label: "Create data visualizations", icon: "📊" },
                              { value: "learn", label: "Learn data science concepts", icon: "📚" },
                            ].map((goal, index) => (
                              <motion.div
                                key={goal.value}
                                className={cn(
                                  "flex items-center space-x-3 rounded-lg border p-3 cursor-pointer transition-colors",
                                  formData.primaryGoal === goal.value 
                                    ? "border-indigo-500/50 bg-indigo-500/10" 
                                    : "border-white/10 bg-white/5 hover:bg-white/10"
                                )}
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                                onClick={() => updateFormData("primaryGoal", goal.value)}
                                initial={{ opacity: 0, x: -10 }}
                                animate={{
                                  opacity: 1,
                                  x: 0,
                                  transition: { delay: 0.1 * index, duration: 0.3 },
                                }}
                              >
                                <RadioGroupItem value={goal.value} id={`goal-${index}`} className="border-white/30" />
                                <span className="text-lg">{goal.icon}</span>
                                <Label htmlFor={`goal-${index}`} className="cursor-pointer w-full text-white/80">
                                  {goal.label}
                                </Label>
                              </motion.div>
                            ))}
                          </RadioGroup>
                        </motion.div>
                        <motion.div variants={fadeInUp} className="space-y-2">
                          <Label htmlFor="targetOutcome" className="text-white/70">
                            What outcome are you hoping to achieve?
                          </Label>
                          <Textarea
                            id="targetOutcome"
                            placeholder="E.g., Predict customer churn, automate reporting, understand trends..."
                            value={formData.targetOutcome}
                            onChange={(e) => updateFormData("targetOutcome", e.target.value)}
                            className="min-h-[80px] bg-white/5 border-white/10 text-white placeholder:text-white/30 focus:border-indigo-500/50"
                          />
                        </motion.div>
                      </CardContent>
                    </>
                  )}

                  {currentStep === 2 && (
                    <>
                      <CardHeader>
                        <div className="flex justify-center mb-2">
                          <div className="w-12 h-12 bg-indigo-500/10 rounded-full flex items-center justify-center">
                            <Briefcase className="w-6 h-6 text-indigo-400" />
                          </div>
                        </div>
                        <CardTitle className="text-white text-center">Professional Background</CardTitle>
                        <CardDescription className="text-white/50 text-center">
                          Tell us about your professional experience
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <motion.div variants={fadeInUp} className="space-y-2">
                          <Label htmlFor="profession" className="text-white/70">What's your profession?</Label>
                          <Input
                            id="profession"
                            placeholder="e.g. Data Analyst, Business Manager, Researcher"
                            value={formData.profession}
                            onChange={(e) => updateFormData("profession", e.target.value)}
                            className="bg-white/5 border-white/10 text-white placeholder:text-white/30 focus:border-indigo-500/50"
                          />
                        </motion.div>

                        <motion.div variants={fadeInUp} className="space-y-2">
                          <Label htmlFor="industry" className="text-white/70">What industry do you work in?</Label>
                          <Select
                            value={formData.industry}
                            onValueChange={(value) => updateFormData("industry", value)}
                          >
                            <SelectTrigger
                              id="industry"
                              className="bg-white/5 border-white/10 text-white focus:border-indigo-500/50"
                            >
                              <SelectValue placeholder="Select an industry" />
                            </SelectTrigger>
                            <SelectContent className="bg-[#1a1a1a] border-white/10">
                              <SelectItem value="technology" className="text-white hover:bg-white/10">Technology</SelectItem>
                              <SelectItem value="finance" className="text-white hover:bg-white/10">Finance & Banking</SelectItem>
                              <SelectItem value="healthcare" className="text-white hover:bg-white/10">Healthcare</SelectItem>
                              <SelectItem value="education" className="text-white hover:bg-white/10">Education</SelectItem>
                              <SelectItem value="retail" className="text-white hover:bg-white/10">Retail & E-commerce</SelectItem>
                              <SelectItem value="manufacturing" className="text-white hover:bg-white/10">Manufacturing</SelectItem>
                              <SelectItem value="consulting" className="text-white hover:bg-white/10">Consulting</SelectItem>
                              <SelectItem value="research" className="text-white hover:bg-white/10">Research & Academia</SelectItem>
                              <SelectItem value="marketing" className="text-white hover:bg-white/10">Marketing & Advertising</SelectItem>
                              <SelectItem value="other" className="text-white hover:bg-white/10">Other</SelectItem>
                            </SelectContent>
                          </Select>
                        </motion.div>

                        <motion.div variants={fadeInUp} className="space-y-2">
                          <Label className="text-white/70">Experience with data science</Label>
                          <RadioGroup
                            value={formData.experience}
                            onValueChange={(value) => updateFormData("experience", value)}
                            className="space-y-2"
                          >
                            {[
                              { value: "beginner", label: "Beginner - Just getting started" },
                              { value: "intermediate", label: "Intermediate - Some experience" },
                              { value: "advanced", label: "Advanced - Experienced practitioner" },
                              { value: "expert", label: "Expert - Professional data scientist" },
                            ].map((level, index) => (
                              <motion.div
                                key={level.value}
                                className={cn(
                                  "flex items-center space-x-3 rounded-lg border p-3 cursor-pointer transition-colors",
                                  formData.experience === level.value 
                                    ? "border-indigo-500/50 bg-indigo-500/10" 
                                    : "border-white/10 bg-white/5 hover:bg-white/10"
                                )}
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                                onClick={() => updateFormData("experience", level.value)}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{
                                  opacity: 1,
                                  y: 0,
                                  transition: { delay: 0.1 * index, duration: 0.3 },
                                }}
                              >
                                <RadioGroupItem value={level.value} id={`exp-${index}`} className="border-white/30" />
                                <Label htmlFor={`exp-${index}`} className="cursor-pointer w-full text-white/80">
                                  {level.label}
                                </Label>
                              </motion.div>
                            ))}
                          </RadioGroup>
                        </motion.div>

                        <AnimatePresence>
                          {error && (
                            <motion.div
                              initial={{ opacity: 0, y: -10 }}
                              animate={{ opacity: 1, y: 0 }}
                              exit={{ opacity: 0 }}
                              className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm"
                            >
                              {error}
                            </motion.div>
                          )}
                          {success && (
                            <motion.div
                              initial={{ opacity: 0, y: -10 }}
                              animate={{ opacity: 1, y: 0 }}
                              exit={{ opacity: 0 }}
                              className="p-3 bg-green-500/10 border border-green-500/20 rounded-lg text-green-400 text-sm"
                            >
                              {success}
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </CardContent>
                    </>
                  )}
                </motion.div>
              </AnimatePresence>

              <CardFooter className="flex justify-between pt-6 pb-4">
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  {currentStep === 0 ? (
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => {
                        setMode('signin');
                        setError(null);
                      }}
                      className="flex items-center gap-1 bg-white/5 hover:bg-white/10 border-white/10 text-white rounded-xl"
                    >
                      <ChevronLeft className="h-4 w-4" /> Sign In
                    </Button>
                  ) : (
                    <Button
                      type="button"
                      variant="outline"
                      onClick={prevStep}
                      className="flex items-center gap-1 bg-white/5 hover:bg-white/10 border-white/10 text-white rounded-xl"
                    >
                      <ChevronLeft className="h-4 w-4" /> Back
                    </Button>
                  )}
                </motion.div>
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <Button
                    type="button"
                    onClick={currentStep === steps.length - 1 ? handleSignUp : nextStep}
                    disabled={!isStepValid() || isSubmitting}
                    className="flex items-center gap-1 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl"
                  >
                    {isSubmitting ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" /> {user ? 'Saving...' : 'Creating...'}
                      </>
                    ) : (
                      <>
                        {currentStep === steps.length - 1 ? (user ? "Complete Profile" : "Create Account") : "Next"}
                        {currentStep === steps.length - 1 ? (
                          <Check className="h-4 w-4" />
                        ) : (
                          <ChevronRight className="h-4 w-4" />
                        )}
                      </>
                    )}
                  </Button>
                </motion.div>
              </CardFooter>
            </div>
          </Card>
        </motion.div>

        <motion.div
          className="mt-4 text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <button
            onClick={onSkip}
            className="text-sm text-white/40 hover:text-white/60 transition-colors"
          >
            Skip for now and continue as guest
          </button>
        </motion.div>

        <motion.div
          className="mt-4 text-center text-sm text-white/40"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          Step {currentStep + 1} of {steps.length}: {steps[currentStep].title}
        </motion.div>
      </div>
    </div>
  );
};

export default AuthPage;
