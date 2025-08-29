-- WARNING: This schema is for context only and is not meant to be run.

CREATE TABLE public.access_logs (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  username text NOT NULL,
  timestamp timestamp without time zone DEFAULT now(),
  CONSTRAINT access_logs_pkey PRIMARY KEY (id)
);
CREATE TABLE public.alerts (
  alert_type text NOT NULL,
  alert_message text,
  severity text,
  timestamp timestamp without time zone DEFAULT now(),
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  sensor_id uuid DEFAULT gen_random_uuid(),
  Device_id text DEFAULT 'ESP32'::text,
  CONSTRAINT alerts_pkey PRIMARY KEY (id),
  CONSTRAINT alerts_sensor_id_fkey FOREIGN KEY (sensor_id) REFERENCES public.sensor_readings(id)
);
CREATE TABLE public.character_emotions (
  character_name text NOT NULL,
  emotion text,
  timestamp timestamp without time zone DEFAULT now(),
  user_id uuid DEFAULT gen_random_uuid(),
  led_color text,
  sound_file text,
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  CONSTRAINT character_emotions_pkey PRIMARY KEY (id),
  CONSTRAINT character_emotions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id)
);
CREATE TABLE public.daily_emotion_percentages (
  day date NOT NULL,
  emotion text NOT NULL,
  percentage numeric,
  CONSTRAINT daily_emotion_percentages_pkey PRIMARY KEY (day, emotion)
);
CREATE TABLE public.emotion_logs (
  emotion text NOT NULL,
  message text DEFAULT ''::text,
  created_at timestamp without time zone DEFAULT now(),
  user_id uuid DEFAULT gen_random_uuid(),
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  CONSTRAINT emotion_logs_pkey PRIMARY KEY (id),
  CONSTRAINT emotion_logs_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id)
);
CREATE TABLE public.sensor_averages (
  sensor_name text NOT NULL,
  avg_value numeric,
  CONSTRAINT sensor_averages_pkey PRIMARY KEY (sensor_name)
);
CREATE TABLE public.sensor_readings (
  sensor_name text NOT NULL,
  value real NOT NULL,
  timestamp timestamp without time zone DEFAULT now(),
  device_id text DEFAULT 'ESP32'::text,
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  CONSTRAINT sensor_readings_pkey PRIMARY KEY (id)
);
CREATE TABLE public.users (
  username text NOT NULL DEFAULT ''::text,
  email text NOT NULL DEFAULT ''::text UNIQUE,
  created_at timestamp without time zone DEFAULT now(),
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  last_login timestamp without time zone,
  CONSTRAINT users_pkey PRIMARY KEY (id)
);
